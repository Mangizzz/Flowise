import { flatten } from 'lodash'
import { Tool } from '@langchain/core/tools'
import { BaseMessage, HumanMessage, AIMessage, SystemMessage } from '@langchain/core/messages'
import { ChainValues } from '@langchain/core/utils/types'
import { RunnableSequence } from '@langchain/core/runnables'
import { BaseChatModel } from '@langchain/core/language_models/chat_models'
import { ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, PromptTemplate } from '@langchain/core/prompts'
import { formatToOpenAIToolMessages } from 'langchain/agents/format_scratchpad/openai_tools'
import { type ToolsAgentStep } from 'langchain/agents/openai/output_parser'
import {
    extractOutputFromArray,
    getBaseClasses,
    handleEscapeCharacters,
    removeInvalidImageMarkdown,
    transformBracesWithColon
} from '../../../src/utils'
import {
    FlowiseMemory,
    ICommonObject,
    INode,
    INodeData,
    INodeParams,
    IServerSideEventStreamer,
    IUsedTool,
    IVisionChatModal
} from '../../../src/Interface'
import { ConsoleCallbackHandler, CustomChainHandler, additionalCallbacks } from '../../../src/handler'
import { AgentExecutor, ToolCallingAgentOutputParser } from '../../../src/agents'
import { Moderation, checkInputs, streamResponse } from '../../moderation/Moderation'
import { formatResponse } from '../../outputparsers/OutputParserHelpers'
import { addImagesToMessages, llmSupportsVision } from '../../../src/multiModalUtils'

// --- Prompt Templates for Each Research Phase ---

const RESEARCH_PLANNER_PROMPT = `You are an expert research planner. Your job is to break down a research topic into a comprehensive research plan.

Given the research topic, create a detailed plan with:
1. **Key Questions**: 3-7 specific questions that must be answered to thoroughly understand the topic
2. **Search Queries**: For each question, provide 2-3 specific search queries that would help find answers
3. **Research Angles**: Different perspectives or dimensions to explore (e.g., technical, economic, social, historical, future trends)

Format your response as:

## Research Plan

### Key Questions
1. [Question 1]
2. [Question 2]
...

### Search Queries
For Q1:
- [Search query 1a]
- [Search query 1b]
For Q2:
- [Search query 2a]
- [Search query 2b]
...

### Research Angles
- [Angle 1]: [Brief description]
- [Angle 2]: [Brief description]
...`

const DEEP_RESEARCHER_PROMPT = `You are a meticulous deep research analyst. You have access to search and web browsing tools to gather comprehensive information.

Your current research task is to investigate the following research plan:
{researchPlan}

## Instructions:
1. Execute the search queries from the plan systematically
2. For each search result, extract key facts, data points, expert opinions, and evidence
3. Look for multiple perspectives and contradicting viewpoints
4. Note the credibility and recency of sources
5. Identify gaps in the information found and do additional searches to fill them
6. Pay special attention to: recent developments, quantitative data, expert consensus, and emerging trends

Be thorough and methodical. Use ALL available tools to gather as much relevant information as possible. Do not stop until you have investigated all the key questions from the research plan.

After your research, compile ALL your raw findings in a structured format with clear source attribution.`

const ANALYSIS_AND_SUMMARY_PROMPT = `You are a senior research analyst who synthesizes raw research data into actionable intelligence reports.

You have received the following raw research findings on the topic: "{topic}"

## Raw Research Findings:
{rawFindings}

## Your Task:
Produce a comprehensive research report with the following structure:

# Deep Research Report: {topic}

## Executive Summary
A concise 2-3 paragraph overview of the most important findings.

## Key Findings
Present the main discoveries organized by theme. Use bullet points and supporting evidence.

## Detailed Analysis
For each major aspect of the topic, provide:
- Current state / Background
- Key data points and evidence
- Different perspectives and viewpoints
- Trends and trajectory

## Recommendations
Provide **specific, actionable recommendations** based on the research. For each recommendation:
1. **What**: Clear statement of the recommendation
2. **Why**: Evidence-based justification from the research
3. **Priority**: High / Medium / Low
4. **Considerations**: Risks, trade-offs, or dependencies

## Sources & Confidence
- List key sources referenced
- Note areas where evidence is strong vs. where more research may be needed

## Conclusion
Final synthesis and forward-looking perspective.

---
Be objective, evidence-based, and highlight where opinions differ from facts. Prioritize actionable insights.`

class DeepResearchAgent_Agents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    inputs: INodeParams[]
    sessionId?: string

    constructor(fields?: { sessionId?: string }) {
        this.label = 'Deep Research Agent'
        this.name = 'deepResearchAgent'
        this.version = 1.0
        this.type = 'AgentExecutor'
        this.category = 'Agents'
        this.icon = 'deepresearch.svg'
        this.description =
            'Autonomous multi-phase research agent that deeply investigates a topic, synthesizes findings, and provides actionable recommendations'
        this.baseClasses = [this.type, ...getBaseClasses(AgentExecutor)]
        this.inputs = [
            {
                label: 'Tools',
                name: 'tools',
                type: 'Tool',
                list: true,
                description: 'Search and web browsing tools for the agent to use (e.g., SerpAPI, Serper, Web Browser, SearchApi)'
            },
            {
                label: 'Memory',
                name: 'memory',
                type: 'BaseChatMemory'
            },
            {
                label: 'Chat Model',
                name: 'model',
                type: 'BaseChatModel',
                description:
                    'The language model to power the research agent. Must support function/tool calling: ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, etc.'
            },
            {
                label: 'Research Depth',
                name: 'researchDepth',
                type: 'options',
                options: [
                    { label: 'Standard (3 research cycles)', name: 'standard', description: '3 research iterations' },
                    { label: 'Deep (5 research cycles)', name: 'deep', description: '5 research iterations' },
                    { label: 'Exhaustive (8 research cycles)', name: 'exhaustive', description: '8 research iterations' }
                ],
                default: 'deep',
                description: 'How many research iterations the agent performs to gather information',
                optional: true
            },
            {
                label: 'Research Focus',
                name: 'researchFocus',
                type: 'string',
                rows: 2,
                placeholder: 'e.g., Focus on technical feasibility and cost-benefit analysis',
                description: 'Optional guidance to focus the research on specific aspects',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Custom System Prompt',
                name: 'systemMessage',
                type: 'string',
                default: `You are a world-class research analyst with expertise in conducting thorough, multi-source investigations. You approach every topic with intellectual curiosity and rigorous methodology.`,
                description: 'Override the default system prompt for the research agent',
                rows: 4,
                optional: true,
                additionalParams: true
            },
            {
                label: 'Max Iterations Per Phase',
                name: 'maxIterations',
                type: 'number',
                default: 15,
                description: 'Maximum tool-calling iterations per research phase',
                optional: true,
                additionalParams: true
            },
            {
                label: 'Input Moderation',
                description: 'Detect text that could generate harmful output and prevent it from being sent to the language model',
                name: 'inputModeration',
                type: 'Moderation',
                optional: true,
                list: true
            }
        ]
        this.sessionId = fields?.sessionId
    }

    async init(nodeData: INodeData, input: string, options: ICommonObject): Promise<any> {
        return prepareResearchAgent(nodeData, options, { sessionId: this.sessionId, chatId: options.chatId, input })
    }

    async run(nodeData: INodeData, input: string, options: ICommonObject): Promise<string | ICommonObject> {
        const model = nodeData.inputs?.model as BaseChatModel
        const memory = nodeData.inputs?.memory as FlowiseMemory
        const moderations = nodeData.inputs?.inputModeration as Moderation[]
        const researchDepth = (nodeData.inputs?.researchDepth as string) || 'deep'
        const researchFocus = (nodeData.inputs?.researchFocus as string) || ''
        const maxIterations = nodeData.inputs?.maxIterations ? parseFloat(nodeData.inputs.maxIterations as string) : 15

        const shouldStreamResponse = options.shouldStreamResponse
        const sseStreamer: IServerSideEventStreamer = options.sseStreamer as IServerSideEventStreamer
        const chatId = options.chatId

        if (moderations && moderations.length > 0) {
            try {
                input = await checkInputs(moderations, input)
            } catch (e) {
                await new Promise((resolve) => setTimeout(resolve, 500))
                if (shouldStreamResponse) {
                    streamResponse(sseStreamer, chatId, e.message)
                }
                return formatResponse(e.message)
            }
        }

        const depthMap: Record<string, number> = { standard: 3, deep: 5, exhaustive: 8 }
        const researchCycles = depthMap[researchDepth] || 5

        let allUsedTools: IUsedTool[] = []
        let allSourceDocuments: ICommonObject[] = []
        let allArtifacts: any[] = []

        // ========================================
        // PHASE 1: Research Planning
        // ========================================
        const planningPrompt = `${RESEARCH_PLANNER_PROMPT}\n\n${researchFocus ? `**Special Focus**: ${researchFocus}\n\n` : ''}Research Topic: ${input}`

        let researchPlan: string
        try {
            const planResult = await model.invoke([new HumanMessage(planningPrompt)])
            researchPlan = typeof planResult.content === 'string' ? planResult.content : JSON.stringify(planResult.content)
        } catch (e) {
            researchPlan = `Research topic: ${input}\nKey questions: What is ${input}? What are the main aspects? What are the pros/cons? What are recommendations?`
        }

        if (shouldStreamResponse && sseStreamer) {
            sseStreamer.streamTokenEvent(chatId, '**Phase 1 - Research Plan Created**\n\n')
        }

        // ========================================
        // PHASE 2: Deep Research Execution
        // ========================================
        if (shouldStreamResponse && sseStreamer) {
            sseStreamer.streamTokenEvent(chatId, '**Phase 2 - Conducting Deep Research...**\n\n')
        }

        const researchSystemPrompt = DEEP_RESEARCHER_PROMPT.replace('{researchPlan}', researchPlan)

        const executor = await prepareResearchAgent(nodeData, options, {
            sessionId: this.sessionId,
            chatId: options.chatId,
            input,
            systemPrompt: researchSystemPrompt,
            maxIterations
        })

        const loggerHandler = new ConsoleCallbackHandler(options.logger)
        const callbacks = await additionalCallbacks(nodeData, options)

        let rawFindings = ''

        // Execute multiple research cycles to gather comprehensive data
        for (let cycle = 0; cycle < researchCycles; cycle++) {
            const cyclePrompt =
                cycle === 0
                    ? `Begin researching the topic: "${input}"\n\nFollow the research plan and start with the first set of questions and search queries. Be thorough and use the available tools.`
                    : `Continue researching. You are on research cycle ${cycle + 1} of ${researchCycles}. ` +
                      `Here is what you've found so far:\n\n${rawFindings}\n\n` +
                      `Now dig deeper: look for information you haven't found yet, explore different angles, ` +
                      `verify key claims, and search for recent developments or contrarian viewpoints. ` +
                      `Focus on filling gaps in the research.`

            try {
                let res: ChainValues
                if (shouldStreamResponse) {
                    const handler = new CustomChainHandler(sseStreamer, chatId)
                    res = await executor.invoke({ input: cyclePrompt }, { callbacks: [loggerHandler, handler, ...callbacks] })
                } else {
                    res = await executor.invoke({ input: cyclePrompt }, { callbacks: [loggerHandler, ...callbacks] })
                }

                if (res.output) {
                    rawFindings += `\n\n--- Research Cycle ${cycle + 1} ---\n${res.output}`
                }
                if (res.sourceDocuments) {
                    allSourceDocuments.push(...flatten(res.sourceDocuments))
                }
                if (res.usedTools) {
                    allUsedTools.push(...flatten(res.usedTools))
                }
                if (res.artifacts) {
                    allArtifacts.push(...flatten(res.artifacts))
                }
            } catch (e) {
                rawFindings += `\n\n--- Research Cycle ${cycle + 1} (partial) ---\nResearch cycle encountered an issue: ${e.message}`
            }
        }

        // ========================================
        // PHASE 3: Analysis and Synthesis
        // ========================================
        if (shouldStreamResponse && sseStreamer) {
            sseStreamer.streamTokenEvent(chatId, '\n\n**Phase 3 - Synthesizing Research & Generating Recommendations...**\n\n')
        }

        const analysisPrompt = ANALYSIS_AND_SUMMARY_PROMPT.replace(/{topic}/g, input).replace('{rawFindings}', rawFindings)

        let finalReport: string
        try {
            if (shouldStreamResponse && sseStreamer) {
                const handler = new CustomChainHandler(sseStreamer, chatId)
                const analysisResult = await model.invoke([new HumanMessage(analysisPrompt)], { callbacks: [handler] })
                finalReport =
                    typeof analysisResult.content === 'string' ? analysisResult.content : JSON.stringify(analysisResult.content)
            } else {
                const analysisResult = await model.invoke([new HumanMessage(analysisPrompt)])
                finalReport =
                    typeof analysisResult.content === 'string' ? analysisResult.content : JSON.stringify(analysisResult.content)
            }
        } catch (e) {
            // Fallback: return the raw findings if analysis fails
            finalReport = `# Research Findings on: ${input}\n\n${rawFindings}\n\n_Note: Automated synthesis was unable to complete. Raw findings are presented above._`
        }

        // Clean up output
        finalReport = removeInvalidImageMarkdown(finalReport)

        // Remove thinking tags (for Claude models)
        const regexPattern: RegExp = /<thinking>[\s\S]*?<\/thinking>/
        const matches: RegExpMatchArray | null = finalReport.match(regexPattern)
        if (matches) {
            for (const match of matches) {
                finalReport = finalReport.replace(match, '')
            }
        }

        // Save to memory
        await memory.addChatMessages(
            [
                {
                    text: input,
                    type: 'userMessage'
                },
                {
                    text: finalReport,
                    type: 'apiMessage'
                }
            ],
            this.sessionId
        )

        // Return with metadata
        if (allSourceDocuments.length || allUsedTools.length || allArtifacts.length) {
            const result: ICommonObject = { text: finalReport }
            if (allSourceDocuments.length) {
                result.sourceDocuments = flatten(allSourceDocuments)
            }
            if (allUsedTools.length) {
                result.usedTools = allUsedTools
            }
            if (allArtifacts.length) {
                result.artifacts = allArtifacts
            }
            return result
        }

        return finalReport
    }
}

const prepareResearchAgent = async (
    nodeData: INodeData,
    options: ICommonObject,
    flowObj: { sessionId?: string; chatId?: string; input?: string; systemPrompt?: string; maxIterations?: number }
) => {
    const model = nodeData.inputs?.model as BaseChatModel
    const maxIterations = flowObj.maxIterations || (nodeData.inputs?.maxIterations ? parseFloat(nodeData.inputs.maxIterations as string) : 15)
    const memory = nodeData.inputs?.memory as FlowiseMemory
    let systemMessage = flowObj.systemPrompt || (nodeData.inputs?.systemMessage as string)
    let tools = nodeData.inputs?.tools
    tools = flatten(tools)
    const memoryKey = memory.memoryKey ? memory.memoryKey : 'chat_history'
    const inputKey = memory.inputKey ? memory.inputKey : 'input'
    const prependMessages = options?.prependMessages

    systemMessage = transformBracesWithColon(systemMessage)

    const prompt = ChatPromptTemplate.fromMessages([
        ['system', systemMessage],
        new MessagesPlaceholder(memoryKey),
        ['human', `{${inputKey}}`],
        new MessagesPlaceholder('agent_scratchpad')
    ])

    if (llmSupportsVision(model)) {
        const visionChatModel = model as IVisionChatModal
        const messageContent = await addImagesToMessages(nodeData, options, model.multiModalOption)

        if (messageContent?.length) {
            visionChatModel.setVisionModel()

            let messagePlaceholder = prompt.promptMessages.pop() as MessagesPlaceholder
            if (prompt.promptMessages.at(-1) instanceof HumanMessagePromptTemplate) {
                const lastMessage = prompt.promptMessages.pop() as HumanMessagePromptTemplate
                const template = (lastMessage.prompt as PromptTemplate).template as string
                const msg = HumanMessagePromptTemplate.fromTemplate([
                    ...messageContent,
                    {
                        text: template
                    }
                ])
                msg.inputVariables = lastMessage.inputVariables
                prompt.promptMessages.push(msg)
            }

            prompt.promptMessages.push(messagePlaceholder)
        } else {
            visionChatModel.revertToOriginalModel()
        }
    }

    if (model.bindTools === undefined) {
        throw new Error(`Deep Research Agent requires a model that supports tool calling (bindTools). Compatible models: ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, ChatMistralAI, ChatVertexAI, GroqChat`)
    }

    const modelWithTools = model.bindTools(tools)

    const runnableAgent = RunnableSequence.from([
        {
            [inputKey]: (i: { input: string; steps: ToolsAgentStep[] }) => i.input,
            agent_scratchpad: (i: { input: string; steps: ToolsAgentStep[] }) => formatToOpenAIToolMessages(i.steps),
            [memoryKey]: async (_: { input: string; steps: ToolsAgentStep[] }) => {
                const messages = (await memory.getChatMessages(flowObj?.sessionId, true, prependMessages)) as BaseMessage[]
                return messages ?? []
            }
        },
        prompt,
        modelWithTools,
        new ToolCallingAgentOutputParser()
    ])

    const executor = AgentExecutor.fromAgentAndTools({
        agent: runnableAgent,
        tools,
        sessionId: flowObj?.sessionId,
        chatId: flowObj?.chatId,
        input: flowObj?.input,
        verbose: process.env.DEBUG === 'true',
        maxIterations
    })

    return executor
}

module.exports = { nodeClass: DeepResearchAgent_Agents }

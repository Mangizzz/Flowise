import { ICommonObject, INode, INodeData, INodeParams } from '../../../src/Interface'
import { getBaseClasses, getCredentialData, getCredentialParam } from '../../../src/utils'
import { desc, FlightPriceSearcherTool, FlightSearchParameters } from './core'

class FlightPriceSearcher_Tools implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]

    constructor() {
        this.label = 'Flight Price Searcher'
        this.name = 'flightPriceSearcher'
        this.version = 1.0
        this.type = 'FlightPriceSearcher'
        this.icon = 'flight.svg'
        this.category = 'Tools'
        this.description = 'Search for flight prices using Amadeus API'
        this.baseClasses = [this.type, ...getBaseClasses(FlightPriceSearcherTool)]
        this.credential = {
            label: 'Connect Credential',
            name: 'credential',
            type: 'credential',
            credentialNames: ['amadeusApi']
        }
        this.inputs = [
            {
                label: 'Description',
                name: 'description',
                type: 'string',
                rows: 4,
                default: desc,
                description: 'Description to tell agent when to use this tool',
                additionalParams: true,
                optional: true
            },
            {
                label: 'Max Results',
                name: 'maxResults',
                type: 'number',
                default: 5,
                description: 'Maximum number of flight offers to return',
                additionalParams: true,
                optional: true
            }
        ]
    }

    async init(nodeData: INodeData, _: string, options: ICommonObject): Promise<any> {
        const credentialData = await getCredentialData(nodeData.credential ?? '', options)
        const amadeusApiKey = getCredentialParam('amadeusApiKey', credentialData, nodeData)
        const amadeusApiSecret = getCredentialParam('amadeusApiSecret', credentialData, nodeData)
        const description = nodeData.inputs?.description as string
        const maxResults = nodeData.inputs?.maxResults as number

        const params: FlightSearchParameters = {
            apiKey: amadeusApiKey,
            apiSecret: amadeusApiSecret
        }

        if (description) params.description = description
        if (maxResults) params.maxResults = maxResults

        return new FlightPriceSearcherTool(params)
    }
}

module.exports = { nodeClass: FlightPriceSearcher_Tools }

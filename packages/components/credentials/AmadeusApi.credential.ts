import { INodeParams, INodeCredential } from '../src/Interface'

class AmadeusApi implements INodeCredential {
    label: string
    name: string
    version: number
    description: string
    inputs: INodeParams[]

    constructor() {
        this.label = 'Amadeus API'
        this.name = 'amadeusApi'
        this.version = 1.0
        this.description = 'Amadeus API credentials for flight search. Get your API key at https://developers.amadeus.com/'
        this.inputs = [
            {
                label: 'API Key',
                name: 'amadeusApiKey',
                type: 'password',
                description: 'Your Amadeus API Key'
            },
            {
                label: 'API Secret',
                name: 'amadeusApiSecret',
                type: 'password',
                description: 'Your Amadeus API Secret'
            }
        ]
    }
}

module.exports = { credClass: AmadeusApi }

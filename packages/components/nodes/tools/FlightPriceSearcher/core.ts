import fetch from 'node-fetch'
import { Tool } from '@langchain/core/tools'

export const desc = `A tool to search for flight prices using the Amadeus API.
Use this when you need to find flight prices between two airports.
Input should be a JSON string with the following properties:
- origin: The IATA airport code for the departure location (e.g., "JFK", "LAX", "LHR")
- destination: The IATA airport code for the arrival location (e.g., "CDG", "SFO", "NRT")
- departureDate: The departure date in YYYY-MM-DD format (e.g., "2024-06-15")
- returnDate: (optional) The return date in YYYY-MM-DD format for round-trip flights
- adults: (optional) Number of adult passengers (default: 1)
- travelClass: (optional) Travel class: ECONOMY, PREMIUM_ECONOMY, BUSINESS, or FIRST (default: ECONOMY)

Example input: {"origin": "JFK", "destination": "LAX", "departureDate": "2024-06-15", "adults": 1}`

export interface FlightSearchParameters {
    apiKey: string
    apiSecret: string
    description?: string
    maxResults?: number
}

interface FlightSearchInput {
    origin: string
    destination: string
    departureDate: string
    returnDate?: string
    adults?: number
    travelClass?: string
}

interface AmadeusTokenResponse {
    access_token: string
    token_type: string
    expires_in: number
}

interface FlightOffer {
    price: {
        total: string
        currency: string
    }
    itineraries: Array<{
        duration: string
        segments: Array<{
            departure: {
                iataCode: string
                at: string
            }
            arrival: {
                iataCode: string
                at: string
            }
            carrierCode: string
            number: string
        }>
    }>
}

export class FlightPriceSearcherTool extends Tool {
    name = 'flight_price_searcher'
    description = desc
    apiKey: string
    apiSecret: string
    maxResults: number
    private accessToken: string | null = null
    private tokenExpiry: number = 0

    constructor(args: FlightSearchParameters) {
        super()
        this.apiKey = args.apiKey
        this.apiSecret = args.apiSecret
        this.description = args.description ?? this.description
        this.maxResults = args.maxResults ?? 5
    }

    private async getAccessToken(): Promise<string> {
        const now = Date.now()
        if (this.accessToken && now < this.tokenExpiry) {
            return this.accessToken
        }

        const tokenUrl = 'https://api.amadeus.com/v1/security/oauth2/token'
        const params = new URLSearchParams()
        params.append('grant_type', 'client_credentials')
        params.append('client_id', this.apiKey)
        params.append('client_secret', this.apiSecret)

        if (process.env.DEBUG === 'true') console.info('Fetching Amadeus access token')

        const response = await fetch(tokenUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: params.toString()
        })

        if (!response.ok) {
            const errorText = await response.text()
            throw new Error(`Failed to get Amadeus access token: ${response.status} - ${errorText}`)
        }

        const data = (await response.json()) as AmadeusTokenResponse
        this.accessToken = data.access_token
        this.tokenExpiry = now + (data.expires_in - 60) * 1000 // Expire 1 minute early to be safe
        return this.accessToken
    }

    private formatDuration(duration: string): string {
        // Convert PT2H30M to "2h 30m"
        const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/)
        if (!match) return duration
        const hours = match[1] ? `${match[1]}h` : ''
        const minutes = match[2] ? `${match[2]}m` : ''
        return `${hours} ${minutes}`.trim()
    }

    private formatDateTime(dateTime: string): string {
        const date = new Date(dateTime)
        return date.toLocaleString('en-US', {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            hour12: true
        })
    }

    private formatFlightOffer(offer: FlightOffer, index: number): string {
        const lines: string[] = []
        lines.push(`\n--- Flight Option ${index + 1} ---`)
        lines.push(`Price: ${offer.price.currency} ${offer.price.total}`)

        offer.itineraries.forEach((itinerary, itinIndex) => {
            const label = offer.itineraries.length > 1 ? (itinIndex === 0 ? 'Outbound' : 'Return') : 'Flight'
            lines.push(`\n${label} (Total Duration: ${this.formatDuration(itinerary.duration)}):`)

            itinerary.segments.forEach((segment, segIndex) => {
                const stops = itinerary.segments.length > 1 ? ` [Segment ${segIndex + 1}/${itinerary.segments.length}]` : ''
                lines.push(
                    `  ${segment.departure.iataCode} -> ${segment.arrival.iataCode}${stops}` +
                        `\n    Departure: ${this.formatDateTime(segment.departure.at)}` +
                        `\n    Arrival: ${this.formatDateTime(segment.arrival.at)}` +
                        `\n    Flight: ${segment.carrierCode}${segment.number}`
                )
            })
        })

        return lines.join('\n')
    }

    async _call(input: string): Promise<string> {
        let searchParams: FlightSearchInput

        try {
            searchParams = JSON.parse(input)
        } catch {
            return 'Error: Invalid input. Please provide a valid JSON string with origin, destination, and departureDate.'
        }

        const { origin, destination, departureDate, returnDate, adults = 1, travelClass = 'ECONOMY' } = searchParams

        if (!origin || !destination || !departureDate) {
            return 'Error: Missing required parameters. Please provide origin, destination, and departureDate.'
        }

        // Validate IATA codes (3 uppercase letters)
        const iataRegex = /^[A-Z]{3}$/
        if (!iataRegex.test(origin.toUpperCase())) {
            return `Error: Invalid origin airport code "${origin}". Please use a valid 3-letter IATA code (e.g., "JFK", "LAX").`
        }
        if (!iataRegex.test(destination.toUpperCase())) {
            return `Error: Invalid destination airport code "${destination}". Please use a valid 3-letter IATA code (e.g., "CDG", "LHR").`
        }

        // Validate date format
        const dateRegex = /^\d{4}-\d{2}-\d{2}$/
        if (!dateRegex.test(departureDate)) {
            return `Error: Invalid departure date format "${departureDate}". Please use YYYY-MM-DD format.`
        }
        if (returnDate && !dateRegex.test(returnDate)) {
            return `Error: Invalid return date format "${returnDate}". Please use YYYY-MM-DD format.`
        }

        try {
            const token = await this.getAccessToken()

            const searchUrl = new URL('https://api.amadeus.com/v2/shopping/flight-offers')
            searchUrl.searchParams.append('originLocationCode', origin.toUpperCase())
            searchUrl.searchParams.append('destinationLocationCode', destination.toUpperCase())
            searchUrl.searchParams.append('departureDate', departureDate)
            if (returnDate) {
                searchUrl.searchParams.append('returnDate', returnDate)
            }
            searchUrl.searchParams.append('adults', adults.toString())
            searchUrl.searchParams.append('travelClass', travelClass.toUpperCase())
            searchUrl.searchParams.append('max', this.maxResults.toString())

            if (process.env.DEBUG === 'true') console.info(`Searching flights: ${searchUrl.toString()}`)

            const response = await fetch(searchUrl.toString(), {
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'application/json'
                }
            })

            if (!response.ok) {
                const errorText = await response.text()
                if (response.status === 400) {
                    return `Error: Invalid search parameters. Please check your airport codes and dates. API response: ${errorText}`
                }
                throw new Error(`Amadeus API error: ${response.status} - ${errorText}`)
            }

            const data = (await response.json()) as { data: FlightOffer[] }

            if (!data.data || data.data.length === 0) {
                return `No flights found from ${origin.toUpperCase()} to ${destination.toUpperCase()} on ${departureDate}${returnDate ? ` with return on ${returnDate}` : ''}.`
            }

            const results: string[] = []
            results.push(`Found ${data.data.length} flight option(s) from ${origin.toUpperCase()} to ${destination.toUpperCase()}:`)

            data.data.forEach((offer, index) => {
                results.push(this.formatFlightOffer(offer, index))
            })

            return results.join('\n')
        } catch (error) {
            if (error instanceof Error) {
                return `Error searching for flights: ${error.message}`
            }
            return 'Error: An unexpected error occurred while searching for flights.'
        }
    }
}

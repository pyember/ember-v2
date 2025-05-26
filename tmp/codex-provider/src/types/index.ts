/**
 * Core types for the provider system
 */

export interface Message {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface CompletionRequest {
  messages: Message[]
  temperature?: number
  maxTokens?: number
  stream?: boolean
}

export interface CompletionResponse {
  content: string
  usage?: {
    input: number
    output: number
  }
}

export interface Provider {
  complete(request: CompletionRequest): Promise<CompletionResponse>
  stream(request: CompletionRequest): AsyncIterable<string>
}
import { Provider } from '../types'

/**
 * Minimal provider registry for managing multiple providers
 */
export class ProviderRegistry {
  private providers = new Map<string, Provider>()
  
  register(name: string, provider: Provider): void {
    this.providers.set(name, provider)
  }
  
  get(name: string): Provider {
    const provider = this.providers.get(name)
    if (!provider) {
      throw new Error(`Unknown provider: ${name}`)
    }
    return provider
  }
  
  has(name: string): boolean {
    return this.providers.has(name)
  }
  
  list(): string[] {
    return Array.from(this.providers.keys())
  }
}
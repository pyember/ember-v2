/**
 * Configuration management utilities.
 * 
 * Follows the pattern of AWS CLI, gcloud, etc:
 * - Credentials stored in ~/.ember/credentials (mode 0600)
 * - Configuration in ~/.ember/config.json
 * - Environment variables take precedence
 */

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';

const CONFIG_DIR = path.join(os.homedir(), '.ember');
const CREDENTIALS_FILE = path.join(CONFIG_DIR, 'credentials');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

// File permissions - readable/writable by owner only (like AWS CLI)
const SECURE_MODE = 0o600;

export interface Credentials {
  [provider: string]: {
    api_key: string;
    created_at: string;
    last_used?: string;
  };
}

export interface Config {
  version: string;
  default_provider?: string;
  providers: {
    [provider: string]: {
      default_model?: string;
      organization_id?: string;
      base_url?: string;
    };
  };
}

/**
 * Save API key securely through Ember's context system.
 */
export async function saveCredentials(provider: string, apiKey: string): Promise<void> {
  // Use Python API to save through context system
  const { execa } = await import('execa');
  
  try {
    await execa('python', [
      '-m', 'ember.cli.commands.configure_api',
      'save-key',
      provider
    ], {
      input: apiKey,
      reject: true
    });
  } catch (error: any) {
    // Fallback to direct file write if Python API fails
    await fs.mkdir(CONFIG_DIR, { recursive: true });
    
    let credentials: Credentials = {};
    try {
      const data = await fs.readFile(CREDENTIALS_FILE, 'utf-8');
      credentials = JSON.parse(data);
    } catch {
      // No existing file
    }
    
    credentials[provider] = {
      api_key: apiKey,
      created_at: new Date().toISOString(),
    };
    
    await fs.writeFile(
      CREDENTIALS_FILE,
      JSON.stringify(credentials, null, 2),
      { mode: SECURE_MODE }
    );
  }
}

/**
 * Load API key from credentials file.
 */
export async function loadCredentials(provider: string): Promise<string | null> {
  try {
    const data = await fs.readFile(CREDENTIALS_FILE, 'utf-8');
    const credentials: Credentials = JSON.parse(data);
    return credentials[provider]?.api_key || null;
  } catch {
    return null;
  }
}

/**
 * Save general configuration through Ember's context system.
 */
export async function saveConfig(updates: Partial<Config>): Promise<void> {
  // Use Python API to save through context system
  const { execa } = await import('execa');
  
  try {
    await execa('python', [
      '-m', 'ember.cli.commands.configure_api',
      'save-config'
    ], {
      input: JSON.stringify(updates),
      reject: true
    });
  } catch (error: any) {
    // Fallback to direct file write if Python API fails
    await fs.mkdir(CONFIG_DIR, { recursive: true });
    
    let config: Config = {
      version: '1.0',
      providers: {}
    };
    
    try {
      const data = await fs.readFile(CONFIG_FILE, 'utf-8');
      config = { ...JSON.parse(data), ...updates };
    } catch {
      config = { ...config, ...updates };
    }
    
    await fs.writeFile(
      CONFIG_FILE,
      JSON.stringify(config, null, 2)
    );
  }
}

/**
 * Check if credentials file exists and is secure.
 */
export async function checkCredentialsSecurity(): Promise<boolean> {
  try {
    const stats = await fs.stat(CREDENTIALS_FILE);
    // Check if only owner can read/write
    return (stats.mode & 0o077) === 0;
  } catch {
    return true; // File doesn't exist yet
  }
}
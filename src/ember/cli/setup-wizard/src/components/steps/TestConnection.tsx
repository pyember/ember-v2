import React, {useEffect, useState} from 'react';
import {Box, Text} from 'ink';
import Spinner from 'ink-spinner';
import {Provider, PROVIDERS} from '../../types.js';
import {execa} from 'execa';
import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';

interface Props {
  provider: Provider;
  apiKey: string;
  onComplete: (success: boolean, message: string) => void;
}

export const TestConnection: React.FC<Props> = ({provider, apiKey, onComplete}) => {
  const [status, setStatus] = useState('Testing connection...');
  const providerInfo = PROVIDERS[provider];

  useEffect(() => {
    const testConnection = async () => {
      try {
        setStatus('Validating API key...');
        
        // Create a safe test script using base64 encoding to avoid shell injection
        const testScript = `
import os
import sys

# Set API key
os.environ['${providerInfo.envVar}'] = '''${apiKey}'''

try:
    from ember.api import models
    response = models("${providerInfo.testModel}", "Say hello!")
    print(f"SUCCESS:{response}")
    sys.exit(0)
except Exception as e:
    print(f"ERROR:{str(e)}", file=sys.stderr)
    sys.exit(1)
`;
        
        // Encode script to avoid shell escaping issues
        const encodedScript = Buffer.from(testScript).toString('base64');
        
        // Execute Python test asynchronously
        const {stdout, stderr, exitCode} = await execa('python', [
          '-c',
          `import base64; exec(base64.b64decode('${encodedScript}').decode('utf-8'))`
        ], {
          env: {...process.env, [providerInfo.envVar]: apiKey},
          reject: false
        });

        if (exitCode === 0 && stdout.startsWith('SUCCESS:')) {
          setStatus('Creating example file...');
          await createExampleFile(provider);
          
          setStatus('Saving configuration...');
          await saveConfiguration(provider, providerInfo.envVar, apiKey);
          
          onComplete(true, 'Setup successful!');
        } else {
          const error = stderr || stdout.replace('ERROR:', '').trim();
          onComplete(false, error || 'Connection failed');
        }
      } catch (error: any) {
        onComplete(false, `Connection failed: ${error.message}`);
      }
    };

    testConnection();
  }, [provider, apiKey, onComplete, providerInfo]);

  return (
    <Box flexDirection="column">
      <Box>
        <Text color="green">
          <Spinner type="dots" />
        </Text>
        <Text> {status}</Text>
      </Box>
      
      <Box marginTop={2}>
        <Text dimColor>This may take a few seconds...</Text>
      </Box>
    </Box>
  );
};

async function createExampleFile(provider: Provider): Promise<void> {
  const providerInfo = PROVIDERS[provider];
  
  // Check if file exists
  let filename = 'hello_ember.py';
  let counter = 1;
  while (await fs.access(filename).then(() => true).catch(() => false)) {
    filename = `hello_ember_${counter}.py`;
    counter++;
  }
  
  // Store model constant name in provider info for consistency
  const modelConstant = providerInfo.testModel.toUpperCase().replace(/[.-]/g, '_');
  
  const example = `#!/usr/bin/env python3
"""First Ember program - demonstrates clean API design."""

from ember.api import models

def main():
    """Simple demonstration of Ember's unified API."""
    model = "${providerInfo.testModel}"
    
    # Example 1: Direct question
    print(f"Model: {model}")
    print(f"Q: What is the capital of France?")
    response = models(model, "What is the capital of France?")
    print(f"A: {response}\\n")
    
    # Example 2: Creative task
    print(f"Q: Write a haiku about programming")
    response = models(model, "Write a haiku about programming")
    print(f"A: {response}\\n")
    
    # Example 3: Conversational assistant
    print("Creating assistant for conversation...")
    assistant = models.instance(model, temperature=0.7)
    
    print(f"Q: Tell me a programming joke")
    response = assistant("Tell me a programming joke")
    print(f"A: {response}\\n")
    
    print(f"Q: Why is it funny?")
    response = assistant("Why is it funny?")
    print(f"A: {response}")

if __name__ == "__main__":
    main()
`;

  await fs.writeFile(filename, example);
  // Make the file executable
  await fs.chmod(filename, 0o755);
}

async function saveConfiguration(provider: Provider, envVar: string, apiKey: string): Promise<void> {
  const { saveCredentials, saveConfig } = await import('../../utils/config.js');
  
  // Save API key securely (like AWS CLI)
  await saveCredentials(provider, apiKey);
  
  // Save general configuration
  await saveConfig({
    providers: {
      [provider]: {
        default_model: PROVIDERS[provider].testModel,
      }
    }
  });
}
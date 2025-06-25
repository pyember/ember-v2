import React, {useState} from 'react';
import {Box, Text, useApp} from 'ink';
import {Welcome} from './steps/Welcome.js';
import {SetupModeSelection} from './steps/SetupModeSelection.js';
import {ProviderSelection} from './steps/ProviderSelection.js';
import {ApiKeySetup} from './steps/ApiKeySetup.js';
import {TestConnection} from './steps/TestConnection.js';
import {Success} from './steps/Success.js';
import {ProgressIndicator} from './ProgressIndicator.js';
import {SetupState, Provider} from '../types.js';

interface Props {
  initialProvider?: Provider;
  contextModel?: string;
  skipWelcome?: boolean;
}

export const SetupWizard: React.FC<Props> = ({
  initialProvider,
  contextModel,
  skipWelcome = false
}) => {
  const {exit} = useApp();
  const [setupSuccess, setSetupSuccess] = useState(false);
  const [state, setState] = useState<SetupState>({
    step: skipWelcome ? (initialProvider ? 'apiKey' : 'setupMode') : 'welcome',
    provider: initialProvider || null,
    apiKey: '',
    testResult: null,
    exampleFile: null,
    setupMode: undefined,
    configuredProviders: new Set<Provider>(),
    remainingProviders: [],
  });

  const handleSetupModeSelect = async (mode: 'single' | 'all') => {
    if (mode === 'all') {
      // Check what's already configured
      const {getConfiguredProviders} = await import('../utils/config.js');
      const configured = await getConfiguredProviders();
      
      const allProviders: Provider[] = ['openai', 'anthropic', 'google'];
      const remaining = allProviders.filter(p => !configured.has(p));
      
      if (remaining.length === 0) {
        // All providers already configured
        setState({...state, setupMode: mode, step: 'success'});
      } else {
        setState({
          ...state, 
          setupMode: mode, 
          configuredProviders: new Set(allProviders.filter(p => configured.has(p)) as Provider[]),
          remainingProviders: remaining,
          provider: remaining[0],
          step: 'apiKey'
        });
      }
    } else {
      setState({...state, setupMode: mode, step: 'provider'});
    }
  };

  const handleProviderSelect = (provider: Provider) => {
    setState({...state, provider, step: 'apiKey'});
  };

  const handleApiKey = (apiKey: string) => {
    setState({...state, apiKey, step: 'test'});
  };
  
  const handleBackToProviderSelection = () => {
    setState({...state, step: 'provider', apiKey: ''});
  };
  
  const handleBackToSetupMode = () => {
    setState({...state, step: 'setupMode', provider: null, apiKey: ''});
  };
  
  const handleSkipProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders && state.remainingProviders.length > 1) {
      // Skip to next provider
      const remaining = state.remainingProviders.slice(1);
      setState({
        ...state,
        remainingProviders: remaining,
        provider: remaining[0],
        apiKey: '',
        step: 'apiKey'
      });
    } else {
      // No more providers, go to success
      setState({...state, step: 'success'});
    }
  };
  
  const handleCancel = () => {
    exit();
  };
  
  const handlePreviousProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders) {
      const allProviders: Provider[] = ['openai', 'anthropic', 'google'];
      const currentIndex = allProviders.indexOf(state.provider!);
      if (currentIndex > 0) {
        setState({
          ...state,
          provider: allProviders[currentIndex - 1],
          apiKey: '',
          step: 'apiKey'
        });
      }
    }
  };
  
  const handleNextProvider = () => {
    if (state.setupMode === 'all' && state.remainingProviders) {
      const allProviders: Provider[] = ['openai', 'anthropic', 'google'];
      const currentIndex = allProviders.indexOf(state.provider!);
      if (currentIndex < allProviders.length - 1) {
        setState({
          ...state,
          provider: allProviders[currentIndex + 1],
          apiKey: '',
          step: 'apiKey'
        });
      }
    }
  };

  const handleTestComplete = (success: boolean, message: string) => {
    if (success) {
      const newConfigured = new Set(state.configuredProviders);
      if (state.provider) {
        newConfigured.add(state.provider);
      }
      
      // Check if we're in multi-provider mode and have more to configure
      if (state.setupMode === 'all' && state.remainingProviders && state.remainingProviders.length > 1) {
        const remaining = state.remainingProviders.slice(1);
        setState({
          ...state,
          configuredProviders: newConfigured,
          remainingProviders: remaining,
          provider: remaining[0],
          apiKey: '',
          step: 'apiKey'
        });
      } else {
        setState({...state, configuredProviders: newConfigured, testResult: {success, message}, step: 'success'});
      }
    } else {
      setState({...state, testResult: {success, message}, step: 'apiKey'});
    }
  };

  const handleComplete = () => {
    exit();
  };

  return (
    <Box flexDirection="column" paddingY={1}>
      {/* Progress indicator for multi-provider setup */}
      {state.setupMode === 'all' && state.step !== 'welcome' && state.step !== 'setupMode' && (
        <Box marginBottom={2}>
          <ProgressIndicator 
            configuredProviders={state.configuredProviders || new Set()}
            currentProvider={state.provider}
          />
        </Box>
      )}
      
      {state.step === 'welcome' && <Welcome onNext={() => setState({...state, step: 'setupMode'})} />}
      {state.step === 'setupMode' && <SetupModeSelection onSelectMode={handleSetupModeSelect} />}
      {state.step === 'provider' && (
        <ProviderSelection 
          onSelect={handleProviderSelect}
          onBack={handleBackToSetupMode}
          onCancel={handleCancel}
        />
      )}
      {state.step === 'apiKey' && (
        <ApiKeySetup 
          provider={state.provider!} 
          onSubmit={handleApiKey}
          onSkip={handleSkipProvider}
          onCancel={handleCancel}
          onPrevious={handlePreviousProvider}
          onNext={handleNextProvider}
          onBack={state.setupMode === 'single' ? handleBackToProviderSelection : undefined}
          error={state.testResult?.success === false ? state.testResult.message : undefined}
          isFirstProvider={state.provider === 'openai'}
          isLastProvider={state.provider === 'google'}
          setupMode={state.setupMode}
        />
      )}
      {state.step === 'test' && <TestConnection provider={state.provider!} apiKey={state.apiKey} onComplete={handleTestComplete} />}
      {state.step === 'success' && <Success provider={state.provider!} onComplete={handleComplete} configuredProviders={state.configuredProviders} />}
    </Box>
  );
};
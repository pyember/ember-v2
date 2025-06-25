import React, {useState} from 'react';
import {Box, Text, useApp} from 'ink';
import {Welcome} from './steps/Welcome.js';
import {ProviderSelection} from './steps/ProviderSelection.js';
import {ApiKeySetup} from './steps/ApiKeySetup.js';
import {TestConnection} from './steps/TestConnection.js';
import {Success} from './steps/Success.js';
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
    step: skipWelcome ? (initialProvider ? 'apiKey' : 'provider') : 'welcome',
    provider: initialProvider || null,
    apiKey: '',
    testResult: null,
    exampleFile: null,
  });

  const handleProviderSelect = (provider: Provider) => {
    setState({...state, provider, step: 'apiKey'});
  };

  const handleApiKey = (apiKey: string) => {
    setState({...state, apiKey, step: 'test'});
  };

  const handleTestComplete = (success: boolean, message: string) => {
    if (success) {
      setState({...state, testResult: {success, message}, step: 'success'});
    } else {
      setState({...state, testResult: {success, message}, step: 'apiKey'});
    }
  };

  const handleComplete = () => {
    exit();
  };

  return (
    <Box flexDirection="column" paddingY={1}>
      {state.step === 'welcome' && <Welcome onNext={() => setState({...state, step: 'provider'})} />}
      {state.step === 'provider' && <ProviderSelection onSelect={handleProviderSelect} />}
      {state.step === 'apiKey' && <ApiKeySetup provider={state.provider!} onSubmit={handleApiKey} error={state.testResult?.success === false ? state.testResult.message : undefined} />}
      {state.step === 'test' && <TestConnection provider={state.provider!} apiKey={state.apiKey} onComplete={handleTestComplete} />}
      {state.step === 'success' && <Success provider={state.provider!} onComplete={handleComplete} />}
    </Box>
  );
};
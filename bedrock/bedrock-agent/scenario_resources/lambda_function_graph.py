def lambda_handler(event, context):
    # https://docs.aws.amazon.com/bedrock/latest/userguide/agents-lambda.html
    agent = event['agent']
    actionGroup = event['actionGroup']
    api_path = event['apiPath']
    # get parameters
    get_parameters = event.get('parameters', [])
    # post parameters
    post_parameters = event['requestBody']['content']['application/json']['properties']

    response_body = {
        'application/json': {
            'body': "sample response"
        }
    }
    
    action_response = {
        'actionGroup': event['actionGroup'],
        'apiPath': event['apiPath'],
        'httpMethod': event['httpMethod'],
        'httpStatusCode': 200,
        'responseBody': response_body
    }
    
    session_attributes = event['sessionAttributes']
    prompt_session_attributes = event['promptSessionAttributes']
    
    api_response = {
        'messageVersion': '1.0', 
        'response': action_response,
        'sessionAttributes': session_attributes,
        'promptSessionAttributes': prompt_session_attributes
    }
        
    return api_response
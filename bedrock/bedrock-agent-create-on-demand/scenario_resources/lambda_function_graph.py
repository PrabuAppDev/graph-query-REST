import json
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Extract necessary information from the event
        agent = event.get('agent')
        actionGroup = event.get('actionGroup')
        apiPath = event.get('apiPath')
        httpMethod = event.get('httpMethod')
        parameters = event.get('parameters', [])
        requestBody = event.get('requestBody', {})
        messageVersion = event.get('messageVersion', '1.0')  # Default to 1.0 if not provided

        # Log the incoming request details
        logger.info(f"Received request on API Path: {apiPath} with Method: {httpMethod}")

        # Here you can insert your business logic or API call
        # For example, interfacing with another AWS service or performing a calculation

        # Build the response body
        responseBody = {
            "application/json": {
                "body": json.dumps({
                    "message": f"The API {apiPath} was called successfully!",
                    "details": f"Called with HTTP method {httpMethod}.",
                    "parameters": parameters,
                    "requestBody": requestBody
                })
            }
        }

        # Construct the action response structure
        action_response = {
            'actionGroup': actionGroup,
            'apiPath': apiPath,
            'httpMethod': httpMethod,
            'httpStatusCode': 200,
            'responseBody': responseBody
        }

        # Build the final API response
        api_response = {
            'response': action_response,
            'messageVersion': messageVersion
        }

        # Log the response
        logger.info("Response: {}".format(json.dumps(api_response)))

        return api_response

    except Exception as e:
        # Log and return an error response if something goes wrong
        logger.error(f"Error processing the request: {str(e)}")
        return {
            'response': {
                'actionGroup': actionGroup,
                'apiPath': apiPath,
                'httpMethod': httpMethod,
                'httpStatusCode': 500,
                'responseBody': {
                    "application/json": {
                        "body": json.dumps({"error": "Internal Server Error"})
                    }
                }
            },
            'messageVersion': messageVersion
        }
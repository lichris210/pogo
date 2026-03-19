#!/bin/bash
# POGO Deployment Script
# Deploys Lambda function + API Gateway

set -e

FUNCTION_NAME="pogo-prompt-generator"
ROLE_NAME="pogo-lambda-role"
API_NAME="pogo-api"
REGION="us-east-1"
RUNTIME="python3.12"
HANDLER="handler.lambda_handler"
TIMEOUT=60
MEMORY=512

echo "=== POGO Deployment ==="

# --- Step 1: Create IAM Role ---
echo ""
echo "Step 1: Creating IAM role..."

TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}'

ROLE_ARN=$(aws iam create-role \
  --role-name $ROLE_NAME \
  --assume-role-policy-document "$TRUST_POLICY" \
  --query 'Role.Arn' \
  --output text 2>/dev/null || \
  aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)

echo "  Role ARN: $ROLE_ARN"

# Attach policies
echo "  Attaching policies..."
aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess 2>/dev/null || true

aws iam attach-role-policy --role-name $ROLE_NAME \
  --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess 2>/dev/null || true

echo "  Waiting 10s for role to propagate..."
sleep 10

# --- Step 2: Package Lambda ---
echo ""
echo "Step 2: Packaging Lambda function..."

# Create a clean package directory
rm -rf /tmp/pogo-package
mkdir -p /tmp/pogo-package

# Copy handler
cp lambda/handler.py /tmp/pogo-package/

# Create zip
cd /tmp/pogo-package
zip -r /tmp/pogo-lambda.zip handler.py
cd -

echo "  Package size: $(du -h /tmp/pogo-lambda.zip | cut -f1)"

# --- Step 3: Deploy Lambda ---
echo ""
echo "Step 3: Deploying Lambda function..."

# Try to create, if exists then update
aws lambda create-function \
  --function-name $FUNCTION_NAME \
  --runtime $RUNTIME \
  --handler $HANDLER \
  --role "$ROLE_ARN" \
  --zip-file fileb:///tmp/pogo-lambda.zip \
  --timeout $TIMEOUT \
  --memory-size $MEMORY \
  --region $REGION \
  2>/dev/null || \
aws lambda update-function-code \
  --function-name $FUNCTION_NAME \
  --zip-file fileb:///tmp/pogo-lambda.zip \
  --region $REGION \
 

echo "  Lambda function deployed!"

# Wait for function to be active
echo "  Waiting for function to be active..."
aws lambda wait function-active --function-name $FUNCTION_NAME --region $REGION

# --- Step 4: Create API Gateway ---
echo ""
echo "Step 4: Creating API Gateway..."

# Create HTTP API
API_ID=$(aws apigatewayv2 create-api \
  --name $API_NAME \
  --protocol-type HTTP \
  --region $REGION \
  --query 'ApiId' \
  --output text 2>/dev/null || \
  aws apigatewayv2 get-apis --region $REGION \
  --query "Items[?Name=='$API_NAME'].ApiId" --output text)

echo "  API ID: $API_ID"

# Get Lambda ARN
LAMBDA_ARN=$(aws lambda get-function \
  --function-name $FUNCTION_NAME \
  --region $REGION \
  --query 'Configuration.FunctionArn' \
  --output text)

# Create integration
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type AWS_PROXY \
  --integration-uri "$LAMBDA_ARN" \
  --payload-format-version "2.0" \
  --region $REGION \
  --query 'IntegrationId' \
  --output text)

echo "  Integration ID: $INTEGRATION_ID"

# Create route
aws apigatewayv2 create-route \
  --api-id $API_ID \
  --route-key "POST /generate" \
  --target "integrations/$INTEGRATION_ID" \
  --region $REGION \
  2>/dev/null || true

# Create default stage with auto-deploy
aws apigatewayv2 create-stage \
  --api-id $API_ID \
  --stage-name '$default' \
  --auto-deploy \
  --region $REGION \
  2>/dev/null || true

# Grant API Gateway permission to invoke Lambda
ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)

aws lambda add-permission \
  --function-name $FUNCTION_NAME \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:$REGION:$ACCOUNT_ID:$API_ID/*" \
  --region $REGION \
  2>/dev/null || true

API_URL="https://$API_ID.execute-api.$REGION.amazonaws.com/generate"

echo ""
echo "============================================"
echo "  POGO DEPLOYED SUCCESSFULLY!"
echo "============================================"
echo ""
echo "  API URL: $API_URL"
echo ""
echo "  Test it:"
echo "  curl -X POST $API_URL \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"task\": \"Analyze customer data\", \"model\": \"claude\"}'"
echo ""
echo "============================================"

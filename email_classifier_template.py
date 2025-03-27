# Configuration and imports
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging
from sklearn.metrics import classification_report # Adding library to get classification metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Sample email dataset
sample_emails = [
    {
        "id": "001",
        "from": "angry.customer@example.com",
        "subject": "Broken product received",
        "body": "I received my order #12345 yesterday but it arrived completely damaged. This is unacceptable and I demand a refund immediately. This is the worst customer service I've experienced.",
        "timestamp": "2024-03-15T10:30:00Z"
    },
    {
        "id": "002",
        "from": "curious.shopper@example.com",
        "subject": "Question about product specifications",
        "body": "Hi, I'm interested in buying your premium package but I couldn't find information about whether it's compatible with Mac OS. Could you please clarify this? Thanks!",
        "timestamp": "2024-03-15T11:45:00Z"
    },
    {
        "id": "003",
        "from": "happy.user@example.com",
        "subject": "Amazing customer support",
        "body": "I just wanted to say thank you for the excellent support I received from Sarah on your team. She went above and beyond to help resolve my issue. Keep up the great work!",
        "timestamp": "2024-03-15T13:15:00Z"
    },
    {
        "id": "004",
        "from": "tech.user@example.com",
        "subject": "Need help with installation",
        "body": "I've been trying to install the software for the past hour but keep getting error code 5123. I've already tried restarting my computer and clearing the cache. Please help!",
        "timestamp": "2024-03-15T14:20:00Z"
    },
    {
        "id": "005",
        "from": "business.client@example.com",
        "subject": "Partnership opportunity",
        "body": "Our company is interested in exploring potential partnership opportunities with your organization. Would it be possible to schedule a call next week to discuss this further?",
        "timestamp": "2024-03-15T15:00:00Z"
    }
]

def expected_categories(email: Dict) -> str: # Function to get the "real" categories
    emails_category = {
        "001": "complaint",
        "002": "inquiry",
        "003": "feedback",
        "004": "support_request",
        "005": "inquiry"
    }

    return emails_category.get(email.get("id"))

def validate_email(email: Dict) -> bool: # Function to validate if an email is appropriate for the process
    required_fields = ["id", "from", "body", "timestamp"]
    
    # Check all required fields
    for field in required_fields:
        if field not in email or not email.get(field):
            logging.warning(f"Missing or empty field: {field}")
            return False
    
    # Content length checks
    if len(email.get("body", "")) == 0:
        logging.warning("Email body is empty")
        return False
    
    if len(email.get("body", "")) > 750:
        logging.warning("Email body too long")
        return False
    
    return True # If no of the conditions is accomplished, then the email is valid


class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

        self.validated_email = None

    def preprocess_email(self, email: Dict) -> Dict: # Function to check if the email is valid to pass through the process
        if validate_email(email):
            self.validated_email = email
            logging.info(f"Validated email {email['id']}")
            return self.validated_email

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        
        TODO: 
        1. Design and implement the classification prompt
        2. Make the API call with appropriate error handling
        3. Validate and return the classification
        """

        # 1
        valid_categories_list = list(self.valid_categories) # To create a list with the valid categories

        classification_prompt = f"""
        Classify the following email into a category and determine the urgency of the matter:
        From: {email.get("from")}
        Subject: {email.get("subject")}
        Body: {email.get("body")}

        Consider the following instructions to classify the email:
        1. The categories to classify the email are: {self.valid_categories}
        2. Analize two main things from the email: 
            - Subject: {email.get("subject")}
            - Body: {email.get("body")}.
            - Also could be useful to analyze who is the sender of the email: {email.get("from")}.
        3. For the specified categories, consider:
            - complaint: Negative feedback, expressing dissatisfaction
            - inquiry: Asking questions about product/service
            - feedback: Positive or neutral comments about experience
            - support_request: Technical issues or help needed
            - other: Does not fit previous categories

        Consider the following instructions to determine the urgency of the email:
        1. Analyze carefully the urgency of the email. Follow the following instructions to set the urgency:
            - If the matter can be attended by our customer support team with standard priority, set is_urgent to False.
            - If there is something that must be done right now that could affect considerably client satisfaction metric or important business relations, set is_urgent to True.

        Finally, if the email is apparently spam or has innapropriate language, set category to 'other' and is_urgent to False.
        """

        # 2
        try:
            self.classification_response = self.client.chat.completions.create( #OpenAI API call for the classification process
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent email classification assistant from a sucessful and busy company, with the capacity to detect the sentiments and intention of the email's content."
                    },
                    {
                        "role": "user",
                        "content": classification_prompt
                    }
                ],
                temperature=0.2, # For precise answer
                max_tokens=15,
                top_p=0.2, # For strict and controlled response
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "email_classificator",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "models assigned category to the email.",
                                    "enum": valid_categories_list
                                },
                                "is_urgent": {
                                    "type": "boolean",
                                    "description": "if the email is urgent to be answered or adviced"
                                }
                            },
                            "required": [
                                "category",
                                "is_urgent"
                            ],
                            "additionalProperties": False
                        }
                    }
                }
            )
            # 3
            json_classification_response = json.loads(self.classification_response.choices[0].message.content)
            if json_classification_response.get("category") not in valid_categories_list:
                json_classification_response["category"] = "other" # If the category set by the model is not in the valid categories, assign other to the field
            return json_classification_response
        except Exception as e:
            logging.error(f"Email classification error: {e}")
            return e
        

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        
        TODO:
        1. Design the response generation prompt
        2. Implement appropriate response templates
        3. Add error handling
        """

        # 2
        # Here I create some mock templates for each of the categories for the model to follow the type of response I want
        complaint_template =  f"""Dear Customer,
        We apologize for the inconvenience you've experienced. We take your concerns seriously and will investigate your issue about "{email.get('subject')}" immediately.
        Our customer service team will contact you soon to resolve this matter.
        Best regards,
        Customer Support Team"""

        inquiry_template = f"""Hello,
        Thank you for reaching out to us with your question about "{email.get('subject')}". We’re happy to assist you.
        A member of our team will review your question and provide a detailed response shortly.
        Best regards,
        Customer Support Team"""

        feedback_template = f"""Dear Valued Customer,
        Thank you for taking the time to share your feedback about "{email.get('subject')}". Your input helps us improve our services and enhance customer satisfaction.
        If there’s anything else you’d like to share, please don’t hesitate to reach out. We truly value your opinion!
        Best regards,
        Customer Support Team"""

        support_template = f"""Hello,
        We've received your support request regarding "{email.get('subject')}". 
        Our technical support team will review the details and provide assistance.
        If you can provide any additional context, it will help us resolve your issue faster.
        Best regards,
        Technical Support Team"""

        other_template = f"""Dear Customer,
        We've received your email and will review it shortly. If your matter requires immediate attention, please contact our support team directly.
        Best regards,
        Customer Support Team"""

        response_templates = {
            "complaint": complaint_template,
            "inquiry": inquiry_template,
            "feedback": feedback_template,
            "support_request": support_template,
            "other": other_template
        }
        # 1
        # Create the prompt for the API call, handling edge cases and how I want the responses
        response_prompt = f"""Create a professional email response for this email:
        From: {email.get("from")}
        Subject: {email.get("subject")}
        Body: {email.get("body")}
        Time: {email.get("timestamp")}

        Generate a concise, empathetic and helpful response, imitating this template:
        {response_templates.get(classification)}

        Do not add customer's name parameter, neither my name parameter.
        Do not answer technical issues, simply tell that our technical team will support the issue.
        Be impartial regarding business, partnerships and relationships.
        Do not specify times or deadlines in your response.
        """
        try:
            self.assisted_response = self.client.chat.completions.create( #OpenAI API call for the response process
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional customer support assistant from a sucessful and busy company."
                    },
                    {
                        "role": "user",
                        "content": response_prompt
                    }
                ],
                temperature=0.3,  # For precise and factual responses
                max_tokens=200,  # Don't want very large responses
                top_p=0.2,  # For strinc and controlled responses
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "email_response_system",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "Response to the email as a professional customer support assistant."
                                }
                            },
                            "required": [
                                "response"
                            ],
                            "additionalProperties": False
                        }
                    }
                }
            )

            json_assisted_response = json.loads(self.assisted_response.choices[0].message.content)
            model_response = json_assisted_response.get("response")
            print("model back response", model_response)
            return model_response
        # 3    
        except Exception as e:
            logging.error(f"Response generation error: {e}")
            return e


class EmailAutomationSystem:
    def __init__(self, processor: EmailProcessor):
        """Initialize the automation system with an EmailProcessor."""
        self.processor = processor
        self.response_handlers = {
            "complaint": self._handle_complaint,
            "inquiry": self._handle_inquiry,
            "feedback": self._handle_feedback,
            "support_request": self._handle_support_request,
            "other": self._handle_other
        }

    def process_email(self, email: Dict) -> Dict:
        """
        Process a single email through the complete pipeline.
        Returns a dictionary with the processing results.
        
        TODO:
        1. Implement the complete processing pipeline
        2. Add appropriate error handling
        3. Return processing results
        """
        # Creating empty initial df
        responses_df = {
            "email_id": email.get("id"),
            "success": False,
            "classification": None,
            "response_sent": None
        }

        # 1
        # Here I execute the automated pipeline stages of classification and responses
        try:
            model_classification = self.processor.classify_email(email)
            category_assigned = model_classification.get("category")
            responses_df["classification"] = category_assigned # Updating the classification df field
            
            self.final_response = self.processor.generate_response(email, category_assigned)
            responses_df["response_sent"] = self.final_response # Updating the response_sent df field

            handler = self.response_handlers.get(category_assigned, self._handle_other) # Specifiyng wich handler to execute
            is_urgent = model_classification.get("is_urgent")
            if is_urgent:
                response_action = create_urgent_ticket(email.get("id"), category_assigned, email.get("body")) # If the matter is urgent, create an urgent ticket to resolve the issue
            else:
                response_action = handler(email) # If not urgent, follow the handler logic from above
            
            responses_df["success"] = True # After complete process, updates the success df field to true
        #2
        except Exception as e:
            logging.error(f"Email processing error for email {email.get('id')}: {e}")
        # 3
        return responses_df # If process fails, returns the empty initial df

    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        # Implement complaint handling logic
        complaint_response_sent = send_complaint_response(email.get("id"), self.final_response) # Send action for complaints
        return complaint_response_sent

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        # Implement inquiry handling logic
        standard_response_sent = send_standard_response(email.get("id"), self.final_response) # Send action for standard issues
        return standard_response_sent

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        customer_feedback_logged = log_customer_feedback(email.get("id"), email.get("body")) # Send action for feedbacks
        return customer_feedback_logged

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        support_ticket_created = create_support_ticket(email.get("id"), email.get("body")) # Send action for support tickets creation
        return support_ticket_created

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        standard_response_sent = send_standard_response(email.get("id"), self.final_response) # Send action for standard issues
        return standard_response_sent

# Mock service functions
def send_complaint_response(email_id: str, response: str):
    """Mock function to simulate sending a response to a complaint"""
    logger.info(f"Sending complaint response for email {email_id}")
    # In real implementation: integrate with email service
    

def send_standard_response(email_id: str, response: str):
    """Mock function to simulate sending a standard response"""
    logger.info(f"Sending standard response for email {email_id}")
    # In real implementation: integrate with email service


def create_urgent_ticket(email_id: str, category: str, context: str):
    """Mock function to simulate creating an urgent ticket"""
    logger.info(f"Creating urgent ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def create_support_ticket(email_id: str, context: str):
    """Mock function to simulate creating a support ticket"""
    logger.info(f"Creating support ticket for email {email_id}")
    # In real implementation: integrate with ticket system


def log_customer_feedback(email_id: str, feedback: str):
    """Mock function to simulate logging customer feedback"""
    logger.info(f"Logging feedback for email {email_id}")
    # In real implementation: integrate with feedback system


def record_metrics(expected_categories: str, predicted_categories: str): # Function to get the classification report
    report = classification_report(expected_categories,predicted_categories, zero_division=0)
    print("\n--- Classification Metrics ---")
    print(report)
    return report


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    real_categories = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        try:
            validated_email = processor.preprocess_email(email) # Validates if the email is appropriate for the process
            if not validated_email:
                logger.warning(f"Email {email['id']} failed validation. Skipping.") # If not, skip that email and continues with the next one
                continue   
            result = automation_system.process_email(email)
            results.append(result)

            expected_category = expected_categories(email)
            real_categories.append(expected_category)

        except Exception as e:
            logger.error(f"Error processing email {email['id']}: {e}")

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])

    classification_metrics = record_metrics(df["classification"].values, real_categories) # execution of the classification report function

    return df, classification_metrics


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()

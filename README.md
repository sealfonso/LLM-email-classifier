# LLM Integration Exercise: Email Classification and Automation

## Overview
In this exercise, you'll build a system that uses Large Language Models (LLMs) to classify incoming emails and automate responses based on the classification. This tests your ability to:
- Integrate LLMs into a Python workflow
- Design and refine prompts for reliable classification
- Implement error handling and reliability measures
- Create automated response systems

## Setup and Requirements

### Environment Setup
1. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Requirements File (requirements.txt)
```
openai>=1.3.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

### Configuration
1. Create a `.env` file in your project root:
```
OPENAI_API_KEY=your_api_key_here
```

## Exercise Structure
The exercise is divided into four main parts:

### Part 1: Email Data Processing (10 minutes)
- Load and validate the provided mock email data
- Create functions to extract email data
- Implement basic data validation

### Part 2: LLM Integration (20 minutes)
- Design classification prompts
- Implement API calls
- Create classification system

### Part 3: Prompt Engineering (20 minutes)
- Analyze classification accuracy
- Identify and handle edge cases
- Document prompt iterations and improvements

### Part 4: Response Automation (10 minutes)
- Create response generation system
- Implement category-specific handling
- Add appropriate error handling and logging

## Starter Code

```python
# Configuration and imports
import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import logging

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


class EmailProcessor:
    def __init__(self):
        """Initialize the email processor with OpenAI API key."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define valid categories
        self.valid_categories = {
            "complaint", "inquiry", "feedback",
            "support_request", "other"
        }

    def classify_email(self, email: Dict) -> Optional[str]:
        """
        Classify an email using LLM.
        Returns the classification category or None if classification fails.
        
        TODO: 
        1. Design and implement the classification prompt
        2. Make the API call with appropriate error handling
        3. Validate and return the classification
        """
        pass

    def generate_response(self, email: Dict, classification: str) -> Optional[str]:
        """
        Generate an automated response based on email classification.
        
        TODO:
        1. Design the response generation prompt
        2. Implement appropriate response templates
        3. Add error handling
        """
        pass


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
        pass

    def _handle_complaint(self, email: Dict):
        """
        Handle complaint emails.
        TODO: Implement complaint handling logic
        """
        pass

    def _handle_inquiry(self, email: Dict):
        """
        Handle inquiry emails.
        TODO: Implement inquiry handling logic
        """
        pass

    def _handle_feedback(self, email: Dict):
        """
        Handle feedback emails.
        TODO: Implement feedback handling logic
        """
        pass

    def _handle_support_request(self, email: Dict):
        """
        Handle support request emails.
        TODO: Implement support request handling logic
        """
        pass

    def _handle_other(self, email: Dict):
        """
        Handle other category emails.
        TODO: Implement handling logic for other categories
        """
        pass

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


def run_demonstration():
    """Run a demonstration of the complete system."""
    # Initialize the system
    processor = EmailProcessor()
    automation_system = EmailAutomationSystem(processor)

    # Process all sample emails
    results = []
    for email in sample_emails:
        logger.info(f"\nProcessing email {email['id']}...")
        result = automation_system.process_email(email)
        results.append(result)

    # Create a summary DataFrame
    df = pd.DataFrame(results)
    print("\nProcessing Summary:")
    print(df[["email_id", "success", "classification", "response_sent"]])

    return df


# Example usage:
if __name__ == "__main__":
    results_df = run_demonstration()
```

## Evaluation Criteria

### 1. Code Quality (25%)
- Clean, well-documented code
- Proper error handling
- Modular design
- Appropriate use of Python best practices

### 2. LLM Integration (25%)
- Effective prompt design
- Proper handling of API calls
- Error handling

### 3. Prompt Engineering (25%)
- Clear documentation of prompt iterations
- Handling of edge cases
- Reasoning about improvements

### 4. System Design (25%)
- Modularity and extensibility
- Logging and monitoring
- Error recovery strategies

## Submission Requirements
1. Your completed code with:
   - Working implementation
   - Documentation of your approach
   - Examples run
2. Documentation of your prompt iterations, including:
   - Initial prompts
   - Problems encountered
   - Improvements made
3. A brief summary covering:
   - Your design decisions
   - Challenges encountered
   - Potential improvements
   - Production considerations

## Notes and Tips
- Start with a simple implementation and iterate
- Document your prompt engineering process
- Consider edge cases in email content
- Test with diverse email examples
- Document any assumptions made

Good luck!

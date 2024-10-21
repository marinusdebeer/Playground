const twilio = require('twilio');
require('dotenv').config();  // Load environment variables

// Ensure the variables are loaded correctly
const accountSid = process.env.TWILIO_ACCOUNT_SID;
const authToken = process.env.TWILIO_AUTH_TOKEN;

console.log(accountSid, authToken);
if (!accountSid || !authToken) {
  throw new Error('Twilio credentials are missing from environment variables.');
}

const client = twilio(accountSid, authToken);

// Send the SMS
client.messages
  .create({
    body: 'Hello from Twilio!',
    from: '+13343098339',  // Your Twilio phone number
    to: '+16472277305'     // Recipient's phone number
  })
  .then(message => console.log(`Message sent with SID: ${message.sid}`))
  .catch(error => console.error('Error sending message:', error));

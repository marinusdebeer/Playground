import Fastify from 'fastify';
import formbody from '@fastify/formbody';
import twilio from 'twilio';
import dotenv from 'dotenv';
import OpenAI from 'openai';

// Load environment variables
dotenv.config();

const { 
  TWILIO_NUMBER, 
  FORWARD_NUMBER, 
  TWILIO_ACCOUNT_SID, 
  TWILIO_AUTH_TOKEN, 
  OPENAI_API_KEY, 
  AI_RESPONSE_ENABLED = 'true' // Default to true if not set
} = process.env;

// Initialize Twilio and OpenAI clients
const twilioClient = twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);
const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

const aiResponseEnabled = AI_RESPONSE_ENABLED.toLowerCase() === 'true'; // Convert string to boolean
console.log(`AI Response Enabled: ${aiResponseEnabled}`);

// Initialize Fastify server
const fastify = Fastify();
fastify.register(formbody); // Register formbody plugin

// System message for OpenAI responses
const SYSTEM_MESSAGE = 'You are taking SMS messages for Zen Zone Cleaning Services. Provide friendly responses, assist with booking, and answer any inquiries about services.';

// --- Utility Functions ---

async function getOpenAIResponse(history) {
  const completion = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: history,
  });
  return completion.choices[0].message['content'];
}

async function getLast10Messages(fromNumber) {
  try {
    const received = await twilioClient.messages.list({ limit: 10, from: fromNumber });
    const sent = await twilioClient.messages.list({ limit: 10, to: fromNumber });

    const allMessages = [...received, ...sent]
      .sort((a, b) => new Date(b.dateSent) - new Date(a.dateSent))
      .slice(0, 20);

    return allMessages.map(msg => ({
      role: msg.from === fromNumber ? 'user' : 'assistant',
      content: msg.body,
    }));
  } catch (error) {
    console.error('Error fetching messages:', error);
    return [];
  }
}

// --- Routes ---

// Route to handle incoming SMS and forward it
fastify.post('/forward-sms', async (request, reply) => {
  const { From, Body } = request.body;

  console.log(`Received SMS from ${From}: ${Body}`);

  try {
    if (aiResponseEnabled) {
      // Fetch message history and generate an AI response
      const messageHistory = await getLast10Messages(From);
      const aiResponse = await getOpenAIResponse([{ role: 'system', content: SYSTEM_MESSAGE }, ...messageHistory]);

      // Send AI-generated response back to the sender
      await twilioClient.messages.create({
        body: aiResponse,
        from: TWILIO_NUMBER,
        to: From,
      });

      console.log(`AI response sent to ${From}: ${aiResponse}`);
    }

    // Forward the original message to the designated number
    await twilioClient.messages.create({
      body: `Forwarded message from ${From}: "${Body}"`,
      from: TWILIO_NUMBER,
      to: FORWARD_NUMBER,
    });

    console.log(`Message forwarded to ${FORWARD_NUMBER}`);

  } catch (error) {
    console.error('Error processing message:', error);
    reply.status(500).send('Internal Server Error');
  }
});

// --- Start the Server ---

const PORT = process.env.PORT || 3000;

fastify.listen({ port: PORT, host: '0.0.0.0' }, (err) => {
  if (err) {
    console.error('Error starting server:', err);
    process.exit(1);
  }
  console.log(`Server running on port ${PORT}`);
});

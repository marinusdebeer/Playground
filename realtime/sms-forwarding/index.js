import Fastify from 'fastify';
import formbody from '@fastify/formbody';
import twilio from 'twilio';
import dotenv from 'dotenv';

dotenv.config();
const { TWILIO_NUMBER, FORWARD_NUMBER, TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN } = process.env;
const twilioClient = twilio(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN);

const fastify = Fastify();

// Register the formbody plugin
fastify.register(formbody);

// Route to handle incoming SMS and forward it
fastify.post('/forward-sms', async (request, reply) => {
  const { From, Body } = request.body;

  console.log(`Received SMS from ${From}: ${Body}`);

  try {
    await twilioClient.messages.create({
      body: `Forwarded message from ${From}: "${Body}"`,
      from: TWILIO_NUMBER,
      to: FORWARD_NUMBER
    });

    console.log(`Message forwarded to ${FORWARD_NUMBER}`);
    // reply.type('text/xml').send('<Response><Message>SMS forwarded successfully.</Message></Response>');
  } catch (error) {
    console.error('Error forwarding message:', error);
    reply.status(500).send('Internal Server Error');
  }
});

const PORT = process.env.PORT || 3000;

fastify.listen({ port: PORT, host: '0.0.0.0' }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`Server running on port ${PORT}`);
});
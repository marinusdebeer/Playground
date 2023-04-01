import express from 'express';
import axios from 'axios';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import cors from 'cors';
import { Configuration, OpenAIApi } from "openai";
dotenv.config();
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,

});
const openai = new OpenAIApi(configuration);


const app = express();
app.use(bodyParser.json());
app.use(cors());

async function createCompletion(prompt) {
  const response = await openai.createCompletion({
    model: "text-davinci-003",
    prompt: prompt,
    temperature: 0.7,
    max_tokens: 400
  });
  const message = response.data.choices[0].text.trim();
  return message
}

app.post('/chat', async (req, res) => {
  // console.log(req.body)
  const { prompt } = req.body;
  if (!prompt) 
    return res.status(400).json({ error: 'Prompt is required' });
  try {
    const message = await createCompletion("write a story")
    console.log(message)
    res.status(200).json({ message });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Error processing the request' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

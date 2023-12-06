import { OpenAI } from 'langchain/llms/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { FaissStore } from 'langchain/vectorstores/faiss';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';

async function initChain() {
  const model = new OpenAI({});
  let loaders = [new PDFLoader('chatbot-with-data/public/ro.pdf')];
  let pages = [];
  for (let loader of loaders) {
    pages.push(...(await loader.load()));
  }
  let text_splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  let docs = text_splitter.splitDocuments(pages);
  let db = await FaissStore.fromDocuments(await docs, new OpenAIEmbeddings());
  /* create vectorstore*/

  return ConversationalRetrievalQAChain.fromLLM(model, db.asRetriever(), {
    returnSourceDocuments: true,
  });
}

export const chain = await initChain();

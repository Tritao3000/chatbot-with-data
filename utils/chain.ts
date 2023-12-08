import { OpenAI } from 'langchain/llms/openai';
import { pinecone } from '@/utils/pinecone-client';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { ConversationalRetrievalQAChain } from 'langchain/chains';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { UnstructuredLoader } from 'langchain/document_loaders/fs/unstructured';
import { STANDALONE_QUESTION_TEMPLATE, QA_TEMPLATE } from './prompt_template';

async function initChain() {
  const model = new OpenAI({
    modelName: 'gpt-3.5-turbo',
    verbose: true,
    temperature: 0,
  });

  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX ?? '');
  console.log('will be loaded');
  let loaders = [new UnstructuredLoader('https://augustalabs.co')];
  console.log('was loaded');
  let pages = [];
  for (let loader of loaders) {
    pages.push(...(await loader.load()));
  }
  console.log('was added');

  let text_splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });
  let docs = text_splitter.splitDocuments(pages);
  console.log('was splitted');

  const documents = await docs;

  // Create vector store
  console.log('docs will be updated');
  await PineconeStore.fromDocuments(documents, new OpenAIEmbeddings({}), {
    pineconeIndex: pineconeIndex,
    namespace: process.env.PINECONE_NAME_SPACE,
    textKey: 'text',
  });
  console.log('docs updated');
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings({}),
    {
      pineconeIndex: pineconeIndex,
      textKey: 'text',
      namespace: process.env.PINECONE_NAME_SPACE,
    }
  );

  return ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    {
      returnSourceDocuments: true,
      qaTemplate: QA_TEMPLATE,
      questionGeneratorTemplate: STANDALONE_QUESTION_TEMPLATE,
    }
  );
}

export const chain = await initChain();

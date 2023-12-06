import { NextResponse } from 'next/server';
import { chain } from '@/utils/chain';
import { Message } from '@/types/message';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const question: string = body.query;
    const history: Message[] = body.history ?? [];

    const res = await chain.call({
      question: question,
      chat_history: history.map((h) => h.content).join('\n'),
    });

    console.log(res.sourceDocuments);

    const links: string[] = Array.from(
      new Set(
        res.sourceDocuments.map(
          (document: { metadata: { source: string } }) =>
            document.metadata.source
        )
      )
    );
    return NextResponse.json({
      role: 'assistant',
      content: res.text,
      links: links,
    });
  } catch (error) {
    console.log('[API_POST]', error);
    return new NextResponse('Internal error', { status: 500 });
  }
}

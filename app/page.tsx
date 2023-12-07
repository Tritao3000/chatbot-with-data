'use client';
import { useState, useRef, useEffect } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Message } from '@/types/message';
import { Send } from 'react-feather';
import LoadingDots from '@/components/LoadingDots';

export default function Home() {
  const [message, setMessage] = useState<string>('');
  const [history, setHistory] = useState<Message[]>([
    {
      role: 'assistant',
      content: 'Hello! Ask me questions about Augusta Labs',
    },
  ]);
  const lastMessageRef = useRef<HTMLDivElement | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  const handleClick = () => {
    if (message == '') return;
    setHistory((oldHistory) => [
      ...oldHistory,
      { role: 'user', content: message },
    ]);
    setMessage('');
    setLoading(true);
    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: message, history: history }),
    })
      .then(async (res) => {
        if (res.ok) {
          const r = await res.json();
          setHistory((oldHistory) => [...oldHistory, r]);
        } else {
          console.error('HTTP error:', res.status);
        }
        setLoading(false);
      })
      .catch((err) => {
        alert(err);
      });
  };

  const formatPageName = (url: string) => {
    // Split the URL by "/" and get the last segment
    const pageName = url.split('/').pop();

    // Split by "-" and then join with space
    if (pageName) {
      const formattedName = pageName.split('-').join(' ');

      // Capitalize only the first letter of the entire string
      return formattedName.charAt(0).toUpperCase() + formattedName.slice(1);
    }
  };

  //scroll to bottom of chat
  useEffect(() => {
    if (lastMessageRef.current) {
      lastMessageRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [history]);

  return (
    <main className="h-screen bg-white p-6 flex flex-col">
      <div className="flex flex-col gap-8 w-full items-center flex-grow max-h-full">
        <h1 className=" text-4xl text-transparent font-extralight bg-clip-text bg-gradient-to-r from-black/80 to-black">
          AugustaGPT
        </h1>
        <form
          className="rounded-2xl border-black border-opacity-5  border lg:w-3/4 flex-grow flex flex-col bg-slate-100 bg-cover max-h-full overflow-clip"
          onSubmit={(e) => {
            e.preventDefault();
            handleClick();
          }}
        >
          <div className="overflow-y-scroll flex flex-col gap-5 p-10 h-full">
            {history.map((message: Message, idx) => {
              const isLastMessage = idx === history.length - 1;
              switch (message.role) {
                case 'assistant':
                  return (
                    <div
                      ref={isLastMessage ? lastMessageRef : null}
                      key={idx}
                      className="flex gap-2"
                    >
                      <Image
                        alt="logo"
                        src="/images/augusta.jpg"
                        width={100}
                        height={100}
                        className="h-12 w-12 rounded-full"
                      />
                      <div className="w-auto max-w-xl break-words bg-white rounded-b-xl rounded-tr-xl text-black p-6 shadow-[0_10px_40px_0px_rgba(0,0,0,0.15)]">
                        <p className="text-sm font-semibold text-black mb-2">
                          AugustaGPT
                        </p>
                        {message.content}
                        {message.links && (
                          <div className="mt-4 flex flex-col gap-2">
                            <p className="text-sm font-medium text-slate-500">
                              Sources:
                            </p>

                            {message.links?.map((link) => {
                              return (
                                <Link
                                  href={link}
                                  key={link}
                                  className="block w-fit px-2 py-1 text-sm  text-violet-700 bg-violet-100 rounded"
                                >
                                  {formatPageName(link)}
                                </Link>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                case 'user':
                  return (
                    <div
                      className="w-auto max-w-xl break-words bg-white rounded-b-xl rounded-tl-xl text-black p-6 self-end shadow-[0_10px_40px_0px_rgba(0,0,0,0.15)]"
                      key={idx}
                      ref={isLastMessage ? lastMessageRef : null}
                    >
                      <p className="text-sm font-semibold text-black mb-2">
                        You
                      </p>
                      {message.content}
                    </div>
                  );
              }
            })}
            {loading && (
              <div ref={lastMessageRef} className="flex gap-2">
                <Image
                  alt="logo"
                  src="/images/augusta.jpg"
                  className="h-12 w-12 rounded-full"
                  width={100}
                  height={100}
                />
                <div className="w-auto max-w-xl break-words bg-white rounded-b-xl rounded-tr-xl text-black p-6 shadow-[0_10px_40px_0px_rgba(0,0,0,0.15)]">
                  <p className="text-sm font-semibold text-black mb-4">
                    AugustaGPT
                  </p>
                  <LoadingDots />
                </div>
              </div>
            )}
          </div>

          {/* input area */}
          <div className="flex sticky bottom-0 w-full px-6 pb-6 h-20">
            <div className="w-full relative ">
              <textarea
                aria-label="chat input"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Type a message"
                className="w-full h-full resize-none rounded-full border border-slate-900/10 bg-white pl-6 pr-24 py-[14px] text-base placeholder:text-slate-400 focus:border-slate-900 focus:outline-none focus:ring-4 focus:ring-slate-500/10 shadow-[0_10px_40px_0px_rgba(0,0,0,0.15)]"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleClick();
                  }
                }}
              />
              <button
                onClick={(e) => {
                  e.preventDefault();
                  handleClick();
                }}
                className="flex w-10 h-10 items-center justify-center rounded-full px-3 text-sm  bg-black font-semibold text-white hover:bg-black/70 active:bg-black/90 absolute right-2 bottom-2 disabled:bg-slate-100 disabled:text-black"
                type="submit"
                aria-label="Send"
                disabled={!message || loading}
              >
                <Send />
              </button>
            </div>
          </div>
        </form>
      </div>
    </main>
  );
}

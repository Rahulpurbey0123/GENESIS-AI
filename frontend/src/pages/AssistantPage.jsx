import React, { useState, useEffect, useRef } from 'react';
import { Bot, Send, User, Sparkles, AlertCircle, ShieldCheck, HelpCircle, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';

export function AssistantPage({ experiment }) {
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [sending, setSending] = useState(false);
  const messagesEndRef = useRef(null);

  const expId = experiment?.id || experiment?.experiment_id;

  useEffect(() => {
    if (expId) {
      setMessages([
        {
          id: 'welcome',
          sender: 'assistant',
          text: `Hello! I am your evidence-grounded AI explanation assistant for experiment '${expId}'. Ask me anything about the model results, DIP profile, search-space reduction, or feature importance.`,
          evidence: null
        }
      ]);
    }
  }, [expId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async (e) => {
    e?.preventDefault();
    if (!inputPrompt.trim() || !expId || sending) return;

    const userMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: inputPrompt
    };

    setMessages((prev) => [...prev, userMessage]);
    const promptToSend = inputPrompt;
    setInputPrompt('');
    setSending(true);

    try {
      const response = await apiService.sendChatMessage(expId, promptToSend);
      const assistantMessage = {
        id: `assistant_${Date.now()}`,
        sender: 'assistant',
        text: response.explanation || 'No explanation generated.',
        evidence: response.evidence_used || null,
        warnings: response.warnings || [],
        provider: response.llm_provider || 'mock',
        model: response.llm_model || 'offline-deterministic',
        intent: response.question_intent || 'GENERAL_EXPERIMENT'
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          sender: 'assistant',
          text: `I don't have enough information in the stored experiment to answer that. (${err.message})`,
          error: true
        }
      ]);
    } finally {
      setSending(false);
    }
  };

  const sampleQuestions = [
    'Why did this model perform well?',
    'What are the most important features?',
    'What does F1 macro mean?',
    'Why was this model recommended?'
  ];

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 space-y-6">
      
      {/* Header */}
      <div className="border-b border-slate-800 pb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
            <Bot className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white">Evidence-Grounded AI Assistant</h2>
            <p className="text-xs text-slate-400">Week 6 Grounded LLM Explanation Service</p>
          </div>
        </div>

        <span className="px-3 py-1 rounded-full bg-sky-500/10 text-sky-300 text-xs font-semibold border border-sky-500/20 flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5 text-sky-400" />
          <span>Audited Grounded Mode</span>
        </span>
      </div>

      {/* Suggested Questions */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-slate-500 self-center font-medium">Suggested:</span>
        {sampleQuestions.map((q, idx) => (
          <button
            key={idx}
            onClick={() => setInputPrompt(q)}
            className="px-3 py-1 rounded-lg bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 text-xs font-medium transition-colors"
          >
            {q}
          </button>
        ))}
      </div>

      {/* Chat Messages Box */}
      <div className="glass-panel rounded-2xl p-6 h-[480px] flex flex-col justify-between space-y-4">
        <div className="overflow-y-auto space-y-4 pr-2">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex items-start gap-3 ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}
            >
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 text-white font-bold text-xs ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600'
                    : 'bg-gradient-to-tr from-sky-600 to-indigo-600'
                }`}
              >
                {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              <div
                className={`max-w-[80%] p-4 rounded-2xl text-xs leading-relaxed space-y-2 ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none'
                }`}
              >
                <p>{msg.text}</p>

                {msg.sender === 'assistant' && msg.provider && (
                  <div className="flex items-center gap-2 pt-1 text-[10px] text-slate-400 border-t border-slate-800/60">
                    <span className="px-1.5 py-0.5 rounded bg-slate-800 font-mono text-sky-300 uppercase">{msg.provider}</span>
                    <span className="font-mono text-slate-400">{msg.model}</span>
                    {msg.intent && (
                      <span className="px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 font-mono">{msg.intent}</span>
                    )}
                  </div>
                )}

                {msg.evidence && (
                  <div className="mt-2 pt-2 border-t border-slate-800 text-[10px] text-slate-400 space-y-1">
                    <span className="font-semibold text-sky-400 block">Grounded Experiment Evidence Used:</span>
                    <pre className="p-2 rounded bg-slate-950/60 font-mono text-[9px] overflow-x-auto text-slate-300">
                      {JSON.stringify(
                        {
                          dataset: msg.evidence.dataset_id || msg.evidence.dataset_name,
                          best_pipeline: msg.evidence.model_name || msg.evidence.best_pipeline?.model_name,
                          metrics: msg.evidence.metrics,
                          metric_score: msg.evidence.model_score
                        },
                        null,
                        2
                      )}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* Prompt Input Form */}
        <form onSubmit={handleSend} className="flex items-center gap-2 pt-2 border-t border-slate-800">
          <input
            type="text"
            value={inputPrompt}
            onChange={(e) => setInputPrompt(e.target.value)}
            placeholder={expId ? "Ask a question about the experiment results..." : "Complete or select an experiment to chat..."}
            disabled={sending || !expId}
            className="flex-1 px-4 py-3 rounded-xl bg-slate-900 border border-slate-700 text-white text-xs outline-none focus:ring-2 focus:ring-sky-500 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={sending || !inputPrompt.trim() || !expId}
            className="px-5 py-3 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs shadow-lg shadow-sky-600/30 flex items-center gap-2 transition-all disabled:opacity-50"
          >
            {sending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            <span>{sending ? 'Explaining...' : 'Ask'}</span>
          </button>
        </form>
      </div>

    </div>
  );
}

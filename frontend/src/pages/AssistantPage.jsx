import React, { useState, useEffect, useRef } from 'react';
import { Bot, Send, User, Sparkles, AlertCircle, ShieldCheck, HelpCircle, Loader2 } from 'lucide-react';
import { apiService } from '../services/api';

export function AssistantPage({ experiment }) {
  const [messages, setMessages] = useState([]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [sending, setSending] = useState(false);
  const [llmStatus, setLlmStatus] = useState(null);
  const messagesEndRef = useRef(null);

  const expId = experiment?.id || experiment?.experiment_id;

  const fetchStatus = async () => {
    try {
      const status = await apiService.getLLMStatus();
      setLlmStatus(status);
    } catch (e) {
      console.warn('Failed to fetch LLM status:', e);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

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

  const executeSend = async (promptToSend) => {
    if (!promptToSend.trim() || !expId || sending) return;

    const userMessage = {
      id: `user_${Date.now()}`,
      sender: 'user',
      text: promptToSend
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputPrompt('');
    setSending(true);

    try {
      const response = await apiService.sendChatMessage(expId, promptToSend);
      if (response.warnings && response.warnings.some(w => w.includes('is missing') || w.includes('Unsupported') || w.includes('error'))) {
        setMessages((prev) => [
          ...prev,
          {
            id: `assistant_${Date.now()}`,
            sender: 'assistant',
            text: response.explanation || 'AI Assistant unavailable.',
            evidence: response.evidence_used || null,
            error: true,
            retryPrompt: promptToSend,
            provider: response.llm_provider || 'mock',
            model: response.llm_model || 'N/A'
          }
        ]);
      } else {
        const assistantMessage = {
          id: `assistant_${Date.now()}`,
          sender: 'assistant',
          text: response.explanation || 'No explanation generated.',
          evidence: response.evidence_used || null,
          warnings: response.warnings || [],
          provider: response.llm_provider || 'mock',
          model: response.llm_model || 'offline-deterministic',
          intent: response.question_intent || 'GENERAL_EXPERIMENT',
          isFallback: Boolean(response.is_fallback || response.validation_status === 'FALLBACK')
        };
        setMessages((prev) => [...prev, assistantMessage]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `err_${Date.now()}`,
          sender: 'assistant',
          text: 'The AI Assistant is temporarily unavailable. The configured LLM provider could not process this request. Please try again.',
          error: true,
          retryPrompt: promptToSend
        }
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleSend = (e) => {
    e?.preventDefault();
    executeSend(inputPrompt);
  };

  const handleRetry = (promptToRetry) => {
    if (promptToRetry) {
      executeSend(promptToRetry);
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
            <p className="text-xs text-slate-400">
              {llmStatus ? (
                llmStatus.provider === 'openrouter' && llmStatus.configured ? (
                  <span className="text-emerald-400 flex items-center gap-1 font-medium">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    ● Real LLM — OpenRouter ({llmStatus.model})
                  </span>
                ) : llmStatus.provider === 'openrouter' && !llmStatus.configured ? (
                  <span className="text-amber-400 flex items-center gap-1 font-medium">
                    <AlertCircle className="w-3.5 h-3.5" />
                    ⚠ OpenRouter not configured
                  </span>
                ) : llmStatus.provider === 'mock' ? (
                  <span className="text-slate-400 flex items-center gap-1 font-medium">
                    <span className="w-2 h-2 rounded-full bg-slate-500"></span>
                    ● Offline Mock Mode
                  </span>
                ) : (
                  <span className="text-rose-400 flex items-center gap-1 font-medium">
                    <AlertCircle className="w-3.5 h-3.5" />
                    ⚠ Unsupported provider configuration
                  </span>
                )
              ) : (
                'Loading LLM Provider Status...'
              )}
            </p>
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
                    : msg.error
                    ? 'bg-rose-600'
                    : 'bg-gradient-to-tr from-sky-600 to-indigo-600'
                }`}
              >
                {msg.sender === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              <div
                className={`max-w-[80%] p-4 rounded-2xl text-xs leading-relaxed space-y-2 ${
                  msg.sender === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none'
                    : msg.error
                    ? 'bg-slate-900 border border-rose-500/40 text-slate-200 rounded-tl-none'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none'
                }`}
              >
                {msg.error ? (
                  <div className="space-y-2">
                    <div className="flex items-center gap-1.5 text-rose-400 font-semibold text-xs">
                      <AlertCircle className="w-4 h-4 shrink-0" />
                      <span>AI Assistant unavailable</span>
                    </div>
                    <p className="text-slate-300">{msg.text}</p>
                    {msg.retryPrompt && (
                      <button
                        onClick={() => handleRetry(msg.retryPrompt)}
                        className="mt-2 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-semibold flex items-center gap-1 transition-colors shadow"
                      >
                        <span>Try Again</span>
                      </button>
                    )}
                  </div>
                ) : (
                  <p>{msg.text}</p>
                )}

                {msg.sender === 'assistant' && msg.provider && !msg.error && (
                  <div className="flex items-center gap-2 pt-1 text-[10px] text-slate-400 border-t border-slate-800/60">
                    {msg.isFallback ? (
                      <span className="px-1.5 py-0.5 rounded bg-amber-950/80 text-amber-300 border border-amber-800/50 font-medium flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-amber-400"></span>
                        Grounded Fallback — Real LLM Unavailable
                      </span>
                    ) : msg.provider === 'openrouter' ? (
                      <span className="px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/50 font-medium flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                        Real LLM — OpenRouter
                      </span>
                    ) : (
                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-medium flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full bg-slate-400"></span>
                        Offline Mock Mode
                      </span>
                    )}
                    <span className="font-mono text-slate-400">{msg.model}</span>
                    {msg.intent && (
                      <span className="px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 font-mono">{msg.intent}</span>
                    )}
                  </div>
                )}

                {msg.evidence && !msg.error && (
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

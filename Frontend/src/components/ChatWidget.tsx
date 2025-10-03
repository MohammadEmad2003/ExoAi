import React, { useState, useRef, useEffect } from "react";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { useTranslation } from "react-i18next";
import {
  MessageCircle,
  Send,
  Bot,
  User,
  Loader2,
  RefreshCw,
  ExternalLink,
  Play,
  Download,
  Upload,
  AlertCircle,
  CheckCircle,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import TechnicalTerm from "@/components/TechnicalTerm";

// Types
interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  language?: string;
}

interface Citation {
  source_id: string;
  snippet: string;
  source_type: string;
  language?: string;
  metadata?: Record<string, any>;
}

interface SuggestedAction {
  label: string;
  action_endpoint: string;
  type: string;
  description?: string;
}

interface ChatResponse {
  reply: string;
  citations: Citation[];
  suggested_actions: SuggestedAction[];
  language: string;
  confidence: number;
}

interface ChatWidgetProps {
  userId?: string;
  sessionId?: string;
  datasetId?: string;
  className?: string;
  onActionTrigger?: (action: SuggestedAction) => void;
  placeholder?: string;
  maxHeight?: string;
}

const ChatWidget: React.FC<ChatWidgetProps> = ({
  userId = "anonymous",
  sessionId,
  datasetId,
  className = "",
  onActionTrigger,
  placeholder,
  maxHeight = "600px",
}) => {
  const { t, i18n } = useTranslation();
  const [kepliImageLoaded, setKepliImageLoaded] = useState(false);
  const [kepliImgSrc, setKepliImgSrc] = useState<string | null>(null);

  // Try a list of possible public paths until an image loads (helps with different dev setups)
  useEffect(() => {
    const candidates = [
      "/Kepli.png",
      "/Copilot_20251004_002122.png",
      "/assets/Copilot_20251004_002122.png",
      "/kepli.png",
    ];

    let mounted = true;

    const testNext = (index: number) => {
      if (!mounted || index >= candidates.length) return;
      const img = new Image();
      img.onload = () => {
        if (!mounted) return;
        setKepliImgSrc(candidates[index]);
        setKepliImageLoaded(true);
      };
      img.onerror = () => {
        testNext(index + 1);
      };
      img.src = candidates[index];
    };

    testNext(0);

    return () => {
      mounted = false;
    };
  }, []);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [currentCitations, setCurrentCitations] = useState<Citation[]>([]);
  const [currentActions, setCurrentActions] = useState<SuggestedAction[]>([]);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const isRTL = i18n.language === "ar";
  const currentLanguage = i18n.language;

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input when widget opens
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: inputValue.trim(),
      timestamp: Date.now(),
      language: currentLanguage,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    setError(null);

    try {
      const ap = import.meta.env.VITE_GEMINI_API_KEY;
      const apiKey = ap as string | undefined;
      const modelName = "gemini-2.5-flash";

      if (!apiKey) {
        throw new Error(
          "Missing VITE_GEMINI_API_KEY or GEMINI_API_KEY in environment"
        );
      }

      const genAI = new GoogleGenerativeAI(apiKey);
      const model = genAI.getGenerativeModel({ model: modelName });
      const kepliPrompt = `
You are Kepli, an AI assistant inside an interactive 3D Universe app.
Your purpose is to help users explore exoplanets and understand astronomy.
You are also a study companion for learning machine learning applied to 
exoplanet detection and classification. Always explain clearly, keep 
answers inspiring and related to space science.

User Question:
${userMessage.content}
`;

      const result = await model.generateContent(kepliPrompt);

      let assistantText =
        (result as any)?.response?.text?.() ||
        (isRTL
          ? "تعذر الحصول على استجابة من النموذج."
          : "Could not get a response from the model.");

      // Remove common LLM self-identification lines to keep Kepli as the assistant
      // e.g. "I do not have a name. I am a large language model..." or "I am a large language model trained by Google."
      assistantText = assistantText
        .split(/\n+/)
        .filter((line: string) => {
          const trimmed = line.trim();
          if (!trimmed) return false;
          const lower = trimmed.toLowerCase();
          // common phrases we want to strip
          const blacklist = [
            "i do not have a name",
            "i do not have a name.",
            "i am a large language model",
            "i am a large language model.",
            "i am a large language model trained by google",
            "i am a large language model trained by google.",
            "i am a large language model trained by google.",
            "i am a model trained by google",
            "i am a model trained by google.",
          ];

          for (const phrase of blacklist) {
            if (lower.includes(phrase)) return false;
          }

          // Arabic common patterns
          if (lower.includes("لا أملك اسم")) return false;
          if (lower.includes("نموذج لغة")) return false;

          return true;
        })
        .join("\n");

      // Add Kepli intro if the model returns a generic identity or doesn't mention Kepli

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: assistantText,
        timestamp: Date.now(),
        language: currentLanguage,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setCurrentCitations([]);
      setCurrentActions([]);
    } catch (err) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error occurred";
      setError(errorMessage);

      // Add error message to chat
      const errorChatMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: isRTL
          ? "عذرًا، حدث خطأ في معالجة رسالتك. يرجى المحاولة مرة أخرى."
          : "Sorry, an error occurred processing your message. Please try again.",
        timestamp: Date.now(),
        language: currentLanguage,
      };

      setMessages((prev) => [...prev, errorChatMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentCitations([]);
    setCurrentActions([]);
    setError(null);
  };

  const handleActionClick = (action: SuggestedAction) => {
    if (onActionTrigger) {
      onActionTrigger(action);
    } else {
      // Default behavior - navigate to endpoint
      window.location.href = action.action_endpoint;
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Intl.DateTimeFormat(currentLanguage, {
      hour: "2-digit",
      minute: "2-digit",
    }).format(new Date(timestamp));
  };

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.role === "user";
    const messageDirection = message.language === "ar" ? "rtl" : "ltr";

    return (
      <div
        key={message.id}
        className={`flex gap-3 mb-4 ${
          isUser ? "flex-row-reverse" : "flex-row"
        } ${isRTL ? "rtl-container" : ""}`}
        dir={messageDirection}
      >
        <div className="flex-shrink-0">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center overflow-hidden ${
              isUser ? "bg-primary text-primary-foreground" : "bg-muted"
            }`}
          >
            {isUser ? (
              <User className="w-4 h-4" />
            ) : kepliImageLoaded && kepliImgSrc ? (
              <img
                src={kepliImgSrc}
                alt="Kepli"
                className="w-8 h-8 object-cover"
                onError={() => setKepliImageLoaded(false)}
              />
            ) : (
              <Bot className="w-4 h-4" />
            )}
          </div>
        </div>

        <div className={`flex-1 ${isUser ? "text-right" : "text-left"}`}>
          <div
            className={`inline-block max-w-[80%] p-3 rounded-lg ${
              isUser ? "bg-primary text-primary-foreground ml-auto" : "bg-muted"
            }`}
          >
            <div className="whitespace-pre-wrap break-words">
              {/* Handle technical terms in assistant messages */}
              {!isUser && message.content.includes("TabKANet") ? (
                <span>
                  {message.content
                    .split(/(TabKANet|QSVC|LightGBM|\.csv|\.json|\.pt|\.onnx)/g)
                    .map((part, index) =>
                      [
                        "TabKANet",
                        "QSVC",
                        "LightGBM",
                        ".csv",
                        ".json",
                        ".pt",
                        ".onnx",
                      ].includes(part) ? (
                        <TechnicalTerm key={index}>{part}</TechnicalTerm>
                      ) : (
                        <span key={index}>{part}</span>
                      )
                    )}
                </span>
              ) : (
                message.content
              )}
            </div>
          </div>

          <div
            className={`text-xs text-muted-foreground mt-1 ${
              isUser ? "text-right" : "text-left"
            }`}
          >
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    );
  };

  const renderCitations = () => {
    if (currentCitations.length === 0) return null;

    return (
      <div className="mt-4 p-3 bg-muted/50 rounded-lg">
        <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
          <ExternalLink className="w-4 h-4" />
          {t("chat.citations")}
        </h4>
        <div className="space-y-2">
          {currentCitations.map((citation, index) => (
            <div
              key={index}
              className="text-xs p-2 bg-background rounded border"
            >
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="outline" className="text-xs">
                  {citation.source_type}
                </Badge>
                {citation.language && citation.language !== currentLanguage && (
                  <Badge variant="secondary" className="text-xs">
                    {citation.language}
                  </Badge>
                )}
              </div>
              <p className="text-muted-foreground">{citation.snippet}</p>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderSuggestedActions = () => {
    if (currentActions.length === 0) return null;

    return (
      <div className="mt-4 p-3 bg-accent/10 rounded-lg">
        <h4 className="text-sm font-medium mb-2">
          {t("chat.suggestedActions")}
        </h4>
        <div className="flex flex-wrap gap-2">
          {currentActions.map((action, index) => (
            <Button
              key={index}
              variant="outline"
              size="sm"
              onClick={() => handleActionClick(action)}
              className="text-xs"
            >
              {action.type === "training" && (
                <Play className="w-3 h-3 mr-1 icon-mirror" />
              )}
              {action.type === "upload" && (
                <Upload className="w-3 h-3 mr-1 icon-mirror" />
              )}
              {action.type === "export" && (
                <Download className="w-3 h-3 mr-1 icon-mirror" />
              )}
              {action.label}
            </Button>
          ))}
        </div>
      </div>
    );
  };

  if (!isOpen) {
    return (
      <div className={`fixed bottom-4 ${isRTL ? "left-4" : "right-4"} z-50`}>
        <Button
          onClick={() => setIsOpen(true)}
          className="rounded-full w-12 h-12 shadow-lg"
          size="icon"
        >
          <MessageCircle className="w-6 h-6" />
        </Button>
      </div>
    );
  }

  return (
    <div
      className={`fixed bottom-4 ${
        isRTL ? "left-4" : "right-4"
      } z-50 ${className}`}
    >
      <Card className="w-96 shadow-lg" style={{ maxHeight }}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg flex items-center gap-2">
              {kepliImageLoaded ? (
                <img
                  src={kepliImgSrc}
                  alt="Kepli"
                  className="w-5 h-5 rounded-full object-cover"
                  onError={() => setKepliImageLoaded(false)}
                />
              ) : (
                <Bot className="w-5 h-5" />
              )}
              {t("chat.title")}
            </CardTitle>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={clearChat}
                title={t("chat.clearHistory")}
                className="w-8 h-8"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                className="w-8 h-8"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {datasetId && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <AlertCircle className="w-4 h-4" />
              {t("chat.contextDataset")}:{" "}
              <TechnicalTerm>{datasetId}</TechnicalTerm>
            </div>
          )}
        </CardHeader>

        <CardContent className="p-0">
          {error && (
            <Alert variant="destructive" className="mx-4 mb-4">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <ScrollArea className="h-96 px-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                {kepliImageLoaded ? (
                  <img
                    src={kepliImgSrc}
                    alt="Kepli"
                    className="w-12 h-12 mb-4 rounded-full opacity-50 object-cover"
                    onError={() => setKepliImageLoaded(false)}
                  />
                ) : (
                  <Bot className="w-12 h-12 mb-4 opacity-50" />
                )}
                <p className="text-sm">{t("chat.welcomeMessage")}</p>
                <p className="text-xs mt-2">{t("chat.helpText")}</p>
              </div>
            ) : (
              <div className="py-4">
                {messages.map(renderMessage)}
                {isLoading && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="text-sm">{t("chat.typing")}</span>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </ScrollArea>

          {(currentCitations.length > 0 || currentActions.length > 0) && (
            <div className="px-4">
              <Separator className="my-2" />
              {renderCitations()}
              {renderSuggestedActions()}
            </div>
          )}

          <div className="p-4 border-t">
            <div className="flex gap-2">
              <Input
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={placeholder || t("chat.inputPlaceholder")}
                disabled={isLoading}
                className={isRTL ? "text-right" : "text-left"}
                dir={isRTL ? "rtl" : "ltr"}
              />
              <Button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading}
                size="icon"
              >
                {isLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4 icon-mirror" />
                )}
              </Button>
            </div>

            <div className="text-xs text-muted-foreground mt-2 text-center">
              {t("chat.poweredBy")} <TechnicalTerm>Gemini AI</TechnicalTerm>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ChatWidget;

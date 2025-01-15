import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid
} from "recharts";
import {
  TrendingUp,
  BarChart2,
  AlertCircle,
  RefreshCw,
  Bell,
  Settings,
  Search,
  Shield,
  Activity
} from "lucide-react";
import DOMPurify from "dompurify";

const COLORS = [
  "#3B82F6", "#10B981", "#F59E0B", "#EF4444",
  "#8B5CF6", "#EC4899", "#14B8A6", "#F43F5E",
  "#8B5CF6", "#06B6D4", "#84CC16", "#2563EB"
];

const API_BASE = "";

// Renders the risk info if riskData is available
const RiskAnalysisCard = ({ riskData }) => {
  if (!riskData) return null;

  const getRiskColor = (level) => {
    const colors = {
      Low: "text-green-600",
      Medium: "text-yellow-600",
      High: "text-red-600"
    };
    return colors[level] || "text-gray-600";
  };

  const MetricCard = ({ title, value, level, Icon }) => (
    <div className="bg-white p-4 rounded-lg shadow-sm border border-gray-100">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="h-4 w-4 text-gray-500" />
        <span className="text-sm text-gray-600">{title}</span>
      </div>
      <div className={`text-lg font-semibold ${getRiskColor(level)}`}>
        {value}
      </div>
    </div>
  );

  return (
    <div className="bg-white rounded-xl shadow-sm p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Shield className="h-5 w-5 text-blue-600" />
          Risk Analysis
        </h2>
        <span className="text-sm text-gray-500">
          Updated: {new Date(riskData.timestamp).toLocaleTimeString()}
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Risk Level"
          value={riskData.risk_level}
          level={riskData.risk_level}
          Icon={AlertCircle}
        />
        <MetricCard
          title="Volatility"
          value={riskData.volatility_level}
          level={riskData.volatility_level}
          Icon={Activity}
        />
        <MetricCard
          title="Performance"
          value={riskData.performance_rating}
          level={riskData.performance_rating}
          Icon={TrendingUp}
        />
        <MetricCard
          title="Sharpe Ratio"
          value={riskData.sharpe_ratio.toFixed(2)}
          level={
            riskData.sharpe_ratio > 2
              ? "Excellent"
              : riskData.sharpe_ratio > 1
              ? "Good"
              : "Poor"
          }
          Icon={BarChart2}
        />
      </div>

      <div className="bg-blue-50 p-4 rounded-lg">
        <h3 className="text-sm font-medium text-blue-900 mb-2">
          Analysis Summary
        </h3>
        <p className="text-sm text-blue-800">{riskData.summary}</p>
      </div>
    </div>
  );
};

function LoadingBar({ loading }) {
  const style = { width: loading ? "70%" : "0%" };
  return <div className="loading-bar" style={style} />;
}

const LoadingIndicator = () => (
  <div className="fixed top-4 right-4 bg-white p-3 rounded-lg shadow-lg flex items-center gap-3 z-50">
    <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />
    <span className="text-gray-700">Processing request...</span>
  </div>
);

function sanitizeAI(text) {
  if (!text) return "";
  const stripped = text.replace(/\*\*/g, "").replace(/\n/g, "<br/>");
  return DOMPurify.sanitize(stripped);
}

export default function App() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showNotifications, setShowNotifications] = useState(false);
  const [membershipMsg, setMembershipMsg] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [riskData, setRiskData] = useState(null);

  // For user news
  const [news, setNews] = useState([]);
  const [expandedNewsIndex, setExpandedNewsIndex] = useState(null);

  // For symbol-specific news
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [symbolNews, setSymbolNews] = useState([]);

  // For AI Chat
  const [conversation, setConversation] = useState([]);
  const [chatInput, setChatInput] = useState("");

  // For tab switching
  const [activeTab, setActiveTab] = useState("news");

  // Filtered users
  const filteredUsers = users.filter((u) =>
    u.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    u.email.toLowerCase().includes(searchTerm.toLowerCase())
  );

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    if (!selectedUser) return;

    // Reset UI state
    setNews([]);
    setSymbolNews([]);
    setSelectedSymbol("");
    setChatInput("");
    setConversation([]);
    setRiskData(null);

    // Build membership msg, fetch news, fetch risk
    buildMembershipMsg(selectedUser.membership_level);
    fetchUserNews(selectedUser.id);
    fetchRiskAnalysis();
  }, [selectedUser]);

  useEffect(() => {
    if (selectedSymbol) {
      fetchSymbolNews(selectedSymbol);
    }
  }, [selectedSymbol]);

  // [1] MAIN: Fetch risk analysis with tickers & weights
  async function fetchRiskAnalysis() {
    if (!selectedUser?.portfolio || selectedUser.portfolio.length === 0) return;

    try {
      // We construct tickers[] and weights[] from the user's portfolio
      const portfolio = selectedUser.portfolio;
      const tickers = portfolio.map(item => item.symbol);
      const totalValue = portfolio.reduce((acc, item) => acc + item.value, 0);
      if (totalValue <= 0) {
        console.warn("Portfolio total value is 0 or negative. Skipping risk call.");
        return;
      }
      const weights = portfolio.map(item => item.value / totalValue);

      const payload = { tickers, weights };

      setLoading(true);
      const res = await fetch(`${API_BASE}/portfolio_risk`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload) // send { tickers, weights } only
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setRiskData(data);
    } catch (err) {
      setError("Failed to load risk analysis");
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function fetchUsers() {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/users`); 
      if (!res.ok) {
        console.error(`HTTP ${res.status} - ${await res.text()}`);
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setUsers(data);
    } catch (err) {
      console.error("Failed to load users:", err);
      setError("Failed to load users: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  // [3] MAIN: Fetch user news
  async function fetchUserNews(userId) {
    try {
      const res = await fetch(`${API_BASE}/users/${userId}/news?limit=5`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data && data.articles) {
        const filtered = data.articles.filter((a) => a.title && !a.title.includes("[Removed]"));
        setNews(filtered);
      } else {
        setNews([]);
      }
    } catch (err) {
      setError("Failed to load user news");
      console.error(err);
    }
  }

  // [4] MAIN: Fetch symbol news
  async function fetchSymbolNews(sym) {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/symbol_news/${sym}?limit=5`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data && data.articles) {
        const filtered = data.articles.filter((a) => a.title && !a.title.includes("[Removed]"));
        setSymbolNews(filtered);
      } else {
        setSymbolNews([]);
      }
    } catch (err) {
      setError("Failed to load stock news");
    } finally {
      setLoading(false);
    }
  }

  // [5] AI chat request
  async function sendAIChatRequest(userMessage) {
    if (!selectedUser) {
      alert("Please select a user first!");
      return;
    }
    setLoading(true);
    try {
      const updatedConversation = [...conversation, { role: "user", content: userMessage }];
      const body = {
        conversation: updatedConversation,
        user_membership: selectedUser.membership_level
      };
      const res = await fetch(`${API_BASE}/users/${selectedUser.id}/ai_chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      const replyMsg = { role: "assistant", content: data.reply };
      const newConversation = [...updatedConversation, replyMsg];
      setConversation(newConversation);
    } catch (err) {
      setError("Failed to get AI chat: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  // [6] AI Chat convenience
  async function getAIAdvice() {
    await sendAIChatRequest("Hi, can you provide a general overview and recommendations for my portfolio?");
  }

  async function askFollowUp() {
    const msg = chatInput.trim();
    if (!msg) return;
    setChatInput("");
    await sendAIChatRequest(msg);
  }

  // [7] Helpers
  function toggleNewsExpand(idx) {
    setExpandedNewsIndex(expandedNewsIndex === idx ? null : idx);
  }

  function getUniqueSymbols() {
    if (!selectedUser?.portfolio) return [];
    const s = new Set(selectedUser.portfolio.map((p) => p.symbol.toUpperCase()));
    return Array.from(s);
  }

  function consolidatePortfolio(portfolio) {
    const map = {};
    for (let p of portfolio) {
      const sym = p.symbol.toUpperCase();
      if (!map[sym]) {
        map[sym] = { symbol: sym, shares: p.shares, value: p.value };
      } else {
        map[sym].shares += p.shares;
        map[sym].value += p.value;
      }
    }
    return Object.values(map);
  }

  function getPortfolioPerformance() {
    return [
      { month: "Jan", value: 30000 },
      { month: "Feb", value: 32000 },
      { month: "Mar", value: 31000 },
      { month: "Apr", value: 34000 },
      { month: "May", value: 33500 },
      { month: "Jun", value: 35000 }
    ];
  }

  function getMembershipColor(level) {
    const colors = {
      gold: "from-yellow-400 to-yellow-600",
      silver: "from-gray-400 to-gray-600",
      basic: "from-blue-400 to-blue-600"
    };
    return colors[level?.toLowerCase()] || colors.basic;
  }

  function buildMembershipMsg(level) {
    const messages = {
      gold: "Premium member: Real-time analysis & priority AI support",
      silver: "Enhanced member: Advanced AI recommendations + up to 5 alerts monthly",
      basic: "Basic member: Weekly analysis and limited AI suggestions"
    };
    setMembershipMsg(messages[level?.toLowerCase()] || messages.basic);
  }

  // Render
  return (
    <div className="min-h-screen bg-gray-50">
      <LoadingBar loading={loading} />
      {loading && <LoadingIndicator />}

      {/* Top Navigation Bar */}
      <div className="w-full bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto">
          <div className="flex justify-between items-center px-4 py-3">
            {/* Brand/Logo */}
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <span className="font-bold text-2xl bg-gradient-to-r from-blue-600 to-indigo-600 text-transparent bg-clip-text">
                FastAgentLM
              </span>
            </div>

            {/* User Controls */}
            <div className="flex items-center gap-6">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search users..."
                  className="pl-10 pr-4 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none w-64"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
                <Search className="h-4 w-4 text-gray-400 absolute left-3 top-3" />
              </div>

              <select
                className="bg-white border border-gray-200 rounded-lg px-4 py-2 text-gray-700 focus:ring-2 focus:ring-blue-500 focus:outline-none appearance-none"
                onChange={(e) => {
                  const val = e.target.value;
                  const userObj = val
                    ? filteredUsers.find((u) => u.id === val)
                    : null;
                  setSelectedUser(userObj);
                }}
                value={selectedUser?.id || ""}
              >
                <option value="">Select User</option>
                {filteredUsers.map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.name} ({u.membership_level})
                  </option>
                ))}
              </select>

              <div className="flex items-center gap-4">
                <button
                  className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                  onClick={() => setShowNotifications(!showNotifications)}
                >
                  <Bell className="h-5 w-5 text-gray-600" />
                </button>
                <button className="p-2 hover:bg-gray-100 rounded-full transition-colors">
                  <Settings className="h-5 w-5 text-gray-600" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      {selectedUser && (
        <div className="max-w-7xl mx-auto px-4 py-6 flex gap-6">
          {/* Left Sidebar */}
          <div className="w-80 flex-shrink-0">
            {/* User Profile Card */}
            <div className="bg-white rounded-xl shadow-sm p-4 mb-4">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-xl font-bold mb-1">{selectedUser.name}</h2>
                  <div
                    className={`px-2 py-1 rounded text-xs font-medium inline-block mb-2 bg-gradient-to-r ${getMembershipColor(
                      selectedUser.membership_level
                    )} text-white`}
                  >
                    {selectedUser.membership_level.toUpperCase()}
                  </div>
                  <p className="text-sm text-gray-600">{membershipMsg}</p>
                </div>
              </div>
            </div>

            {/* News and Chat Tabs */}
            <div className="bg-white rounded-xl shadow-sm overflow-hidden">
              <div className="flex border-b border-gray-200">
                <button
                  className={`flex-1 px-4 py-3 text-sm font-medium ${
                    activeTab === "news"
                      ? "text-blue-600 border-b-2 border-blue-600"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                  onClick={() => setActiveTab("news")}
                >
                  News
                </button>
                <button
                  className={`flex-1 px-4 py-3 text-sm font-medium ${
                    activeTab === "chat"
                      ? "text-blue-600 border-b-2 border-blue-600"
                      : "text-gray-600 hover:text-gray-900"
                  }`}
                  onClick={() => setActiveTab("chat")}
                >
                  AI Chat
                </button>
              </div>

              <div className="p-4">
                {activeTab === "news" ? (
                  <div className="space-y-4">
                    <div className="flex justify-between items-center mb-2">
                      <h3 className="font-medium">Latest Updates</h3>
                      <button
                        onClick={() => fetchUserNews(selectedUser.id)}
                        className="text-blue-600 hover:text-blue-700 flex items-center gap-1 text-sm"
                      >
                        <RefreshCw className="h-4 w-4" />
                        Refresh
                      </button>
                    </div>
                    <div className="max-h-[600px] overflow-y-auto custom-scrollbar space-y-3">
                      {news.map((article, idx) => (
                        <div
                          key={idx}
                          className="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition cursor-pointer"
                          onClick={() => setExpandedNewsIndex(
                            expandedNewsIndex === idx ? null : idx
                          )}
                        >
                          <h4 className="font-medium text-sm">
                            {article.title}
                          </h4>
                          {expandedNewsIndex === idx && (
                            <p className="text-xs text-gray-600 mt-1">
                              {article.description}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <button
                      onClick={getAIAdvice}
                      className="w-full bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center justify-center gap-2 text-sm font-medium"
                    >
                      <BarChart2 className="h-4 w-4" />
                      Get AI Analysis
                    </button>

                    <div className="max-h-[400px] overflow-y-auto custom-scrollbar space-y-3">
                      {conversation.map((msg, idx) => (
                        <div
                          key={idx}
                          className={`p-3 rounded-lg ${
                            msg.role === "assistant"
                              ? "bg-gray-50 text-gray-900"
                              : "bg-blue-50 text-blue-900"
                          }`}
                        >
                          <div
                            dangerouslySetInnerHTML={{
                              __html: sanitizeAI(msg.content),
                            }}
                          />
                        </div>
                      ))}
                    </div>

                    <div className="flex gap-2">
                      <input
                        type="text"
                        className="flex-1 px-3 py-2 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Ask a question..."
                        value={chatInput}
                        onChange={(e) => setChatInput(e.target.value)}
                      />
                      <button
                        onClick={askFollowUp}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700"
                      >
                        Send
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="flex-1 space-y-6">
            {/* Portfolio Performance Chart */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <h2 className="text-lg font-semibold mb-4">Portfolio Performance</h2>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={getPortfolioPerformance()}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
                    <XAxis dataKey="month" stroke="#6B7280" />
                    <YAxis stroke="#6B7280" />
                    <Tooltip />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#3B82F6"
                      fillOpacity={1}
                      fill="url(#colorValue)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Risk Analysis Section */}
            {riskData && <RiskAnalysisCard riskData={riskData} />}

            {/* Holdings Distribution */}
            <div className="bg-white rounded-xl shadow-sm p-6">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold">Holdings Distribution</h2>
                <div className="flex gap-2">
                  {selectedUser.portfolio &&
                    getUniqueSymbols().map((sym) => (
                      <button
                        key={sym}
                        onClick={() => setSelectedSymbol(sym)}
                        className={`px-3 py-1 rounded-lg text-sm ${
                          selectedSymbol === sym
                            ? "bg-blue-100 text-blue-700 font-medium"
                            : "bg-gray-100 text-gray-700 hover:bg-gray-200"
                        }`}
                      >
                        {sym}
                      </button>
                    ))}
                </div>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={consolidatePortfolio(selectedUser.portfolio)}
                      dataKey="value"
                      nameKey="symbol"
                      outerRadius={80}
                      innerRadius={60}
                    >
                      {consolidatePortfolio(selectedUser.portfolio).map(
                        (entry, i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        )
                      )}
                    </Pie>
                    <Tooltip formatter={(val) => `$${val.toLocaleString()}`} />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Symbol-specific News */}
            {selectedSymbol && symbolNews.length > 0 && (
              <div className="bg-white rounded-xl shadow-sm p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">{selectedSymbol} News</h2>
                  <button
                    onClick={() => fetchSymbolNews(selectedSymbol)}
                    className="text-blue-600 hover:text-blue-700 flex items-center gap-2"
                  >
                    <RefreshCw className="h-4 w-4" />
                    Refresh
                  </button>
                </div>
                <div className="space-y-4 max-h-[400px] overflow-y-auto custom-scrollbar">
                  {symbolNews.map((article, idx) => (
                    <div
                      key={idx}
                      className="p-4 border border-gray-100 rounded-lg hover:border-blue-200 transition-all duration-200"
                    >
                      <a
                        href={article.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="block group"
                      >
                        <h3 className="font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                          {article.title}
                        </h3>
                        <p className="text-sm text-gray-600 mt-1">
                          {article.description}
                        </p>
                      </a>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

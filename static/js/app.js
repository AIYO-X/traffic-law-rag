/* 前端交互：SSE 流式输出 + 多轮对话 + 检索来源展示 + 可打断生成 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const messagesEl = $("messages");
  const form = $("chatForm");
  const input = $("questionInput");
  const sendBtn = $("sendBtn");
  const stopBtn = $("stopBtn");
  const sourceList = $("sourceList");
  const statusBadge = $("statusBadge");
  const statusLine = $("statusLine");

  let sessionId = localStorage.getItem("traffic_session_id") || "";
  if (!sessionId) {
    sessionId = "s-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
    localStorage.setItem("traffic_session_id", sessionId);
  }

  let busy = false;              // 是否正在生成
  let controller = null;         // 当前 SSE 请求的 AbortController
  let requestSeq = 0;            // 请求代次：旧请求的收尾不得覆盖新请求状态
  let currentBubble = null;      // 当前助手气泡（停止时追加标记）
  let pendingQueue = [];         // 待处理问题队列：生成中提问时排队，等当前回答完成后再问

  function escapeHtml(text) {
    return String(text == null ? "" : text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // 极简 Markdown 渲染：粗体、行内代码、引用编号、换行
  function renderMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\[(\d+)\]/g, '<span class="cite" data-cite="$1">[$1]</span>');
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function scrollBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function addMessage(role, text) {
    const wrap = document.createElement("div");
    wrap.className = "message " + role;
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "我" : "法";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = renderMarkdown(text);
    wrap.appendChild(avatar);
    wrap.appendChild(bubble);
    messagesEl.appendChild(wrap);
    scrollBottom();
    return bubble;
  }

  function setStatus(text) {
    statusLine.textContent = text || "";
  }

  function renderSources(sources) {
    if (!sources || !sources.length) {
      sourceList.innerHTML = '<p class="empty">知识库中没有检索到与这个问题直接相关的片段，回答可能依据不足。</p>';
      return;
    }
    sourceList.innerHTML = sources.map(function (item, index) {
      const score = typeof item.score === "number" ? (item.score * 100).toFixed(1) : "—";
      return '<article class="source-card" id="source-' + (index + 1) + '">' +
        '<div class="source-head">' +
          '<span class="source-index">[' + (index + 1) + ']</span>' +
          '<span class="source-score">相似度 ' + score + '%</span>' +
        '</div>' +
        '<h3>' + escapeHtml(item.title) + '</h3>' +
        '<p class="source-meta">' + escapeHtml(item.source) + '</p>' +
        '<p class="source-content">' + escapeHtml(item.content) + '</p>' +
        '<span class="source-tag">' + escapeHtml(item.category) + '</span>' +
      '</article>';
    }).join("");
  }

  function renderOffTopicSources() {
    sourceList.innerHTML =
      '<div class="off-topic-note">' +
        '<strong>本次未进行法规检索</strong>' +
        '<p>这个问题似乎和交通出行法规没什么关系，我就没有去翻法规库啦～ 换一个出行、驾驶或交通违法相关的疑问，我会帮你把规定和出处找出来。</p>' +
      '</div>';
  }

  // 在气泡尾部追加「已停止生成」标记，并移除加载中的转圈提示
  function appendStoppedNote(bubble, note) {
    if (!bubble || bubble.querySelector(".stopped-note")) return;
    const typing = bubble.querySelector(".typing");
    if (typing) typing.remove();
    const span = document.createElement("span");
    span.className = "stopped-note";
    span.textContent = note || "已停止生成";
    bubble.appendChild(span);
  }

  function setBusyState(on) {
    busy = on;
    // 输入框与发送按钮始终可用：生成中也可以继续输入（新问题会排队，等当前回答完成后再答）
    stopBtn.classList.toggle("visible", on);
    document.body.classList.toggle("is-generating", on);
    sendBtn.title = on ? "发送新问题会排队，当前回答完成后继续" : "发送";
  }

  // 中断当前生成：前端中止 SSE，同时通知后端停止大模型推理
  function abortCurrent(note) {
    if (controller) {
      try { controller.abort(); } catch (e) { /* 已结束 */ }
      controller = null;
    }
    try {
      fetch("/api/stop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
        keepalive: true
      }).catch(function () { /* 通知失败不影响前端状态 */ });
    } catch (e) { /* 忽略 */ }

    if (currentBubble) {
      appendStoppedNote(currentBubble, note);
      currentBubble = null;
    }
    requestSeq += 1;          // 使旧请求的 finally 收尾失效
    setBusyState(false);
  }

  function handleEvent(event, bubble, state) {
    switch (event.event) {
      case "status":
        setStatus(event.data);
        break;
      case "sources":
        renderSources(event.data);
        break;
      case "token":
        state.answer += event.data || "";
        bubble.innerHTML = renderMarkdown(state.answer);
        scrollBottom();
        break;
      case "stopped":
        appendStoppedNote(bubble, "已停止生成");
        setStatus("已停止生成，可以继续提问");
        break;
      case "done":
        state.answer = event.data.answer || state.answer;
        bubble.innerHTML = renderMarkdown(state.answer);
        if (event.data.off_topic) {
          bubble.classList.add("off-topic");
          renderOffTopicSources();
          setStatus("已温和引导回交通法规话题");
        } else {
          setStatus("回答完成 · 引用 " + (event.data.sources || []).length + " 条法规");
        }
        scrollBottom();
        break;
      case "error":
        bubble.innerHTML = '<span class="error-text">' + escapeHtml(event.data) + "</span>";
        setStatus("");
        break;
      default:
        break;
    }
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question) return;

    // 生成中提出新问题：不打断当前回答，而是排队，等当前回答完成后依次处理
    if (busy) {
      pendingQueue.push(question);
      setStatus("已排队 1 个问题，当前回答完成后将继续（还有 " + pendingQueue.length + " 个待回答）");
      return;
    }

    const mySeq = ++requestSeq;
    const myController = new AbortController();
    controller = myController;
    setBusyState(true);

    addMessage("user", question);
    const bubble = addMessage("assistant", "");
    currentBubble = bubble;
    bubble.innerHTML = '<span class="typing">正在检索法规知识库</span>';
    setStatus("正在连接本地模型…");

    const state = { answer: "" };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question, session_id: sessionId }),
        signal: myController.signal
      });

      const contentType = response.headers.get("Content-Type") || "";
      if (!response.ok) {
        let message = "请求失败（" + response.status + "）";
        try {
          const body = await response.json();
          message = body.error || message;
        } catch (e) { /* 忽略非 JSON 错误体 */ }
        throw new Error(message);
      }
      if (contentType.indexOf("text/event-stream") === -1) {
        const body = await response.json().catch(function () { return {}; });
        throw new Error(body.error || "服务端返回格式异常，请检查后端日志。");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const result = await reader.read();
        if (result.done) break;
        if (mySeq !== requestSeq) break;   // 已被更新的请求接管
        buffer += decoder.decode(result.value, { stream: true });

        let position;
        while ((position = buffer.indexOf("\n\n")) !== -1) {
          const raw = buffer.slice(0, position);
          buffer = buffer.slice(position + 2);
          const dataLine = raw.split("\n").find(function (line) {
            return line.indexOf("data: ") === 0;
          });
          if (!dataLine) continue;
          let event;
          try {
            event = JSON.parse(dataLine.slice(6));
          } catch (e) {
            continue;
          }
          handleEvent(event, bubble, state);
        }
      }
      if (mySeq === requestSeq && !state.answer) {
        bubble.innerHTML = '<span class="error-text">模型没有返回内容，请检查 Ollama 服务与模型状态。</span>';
      }
    } catch (error) {
      if (mySeq !== requestSeq) return;             // 旧请求被接管，忽略其错误
      if (error && error.name === "AbortError") return; // 主动停止，不显示错误
      bubble.innerHTML = '<span class="error-text">' + escapeHtml(error.message || error) + "</span>";
    } finally {
      // 只有「最新一代」请求才能收尾，避免旧请求覆盖新一轮的界面状态
      if (mySeq === requestSeq) {
        setBusyState(false);
        controller = null;
        currentBubble = null;
        input.focus();
        // 当前回答已完整结束，若有排队的问题，取出队首继续作答
        if (pendingQueue.length) {
          const next = pendingQueue.shift();
          ask(next);
        }
      }
    }
  }

  // 停止生成按钮：立即中断，输入框保持可用
  stopBtn.addEventListener("click", function () {
    if (!busy) return;
    abortCurrent("已停止生成");
    setStatus("已停止生成，可以继续提问");
    input.focus();
  });

  // 引用编号点击 → 滚动到对应来源卡片
  messagesEl.addEventListener("click", function (event) {
    const cite = event.target.closest(".cite");
    if (!cite) return;
    const target = document.getElementById("source-" + cite.dataset.cite);
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.add("highlight");
    setTimeout(function () { target.classList.remove("highlight"); }, 1800);
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    const question = input.value;
    if (!question.trim()) return;   // 生成中也允许发送（会打断当前回答）
    input.value = "";
    ask(question);
  });

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll(".quick-questions button").forEach(function (button) {
    button.addEventListener("click", function () {
      ask(button.dataset.q);
    });
  });

  $("clearBtn").addEventListener("click", async function () {
    if (busy) abortCurrent("对话已清空");
    pendingQueue = [];          // 清空对话时同时丢弃排队中的问题
    try {
      await fetch("/api/clear", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId })
      });
    } catch (e) { /* 清空失败也重置界面 */ }
    messagesEl.innerHTML = '<div class="welcome"><h2>对话已清空</h2>' +
      '<p>上下文已重置，现在可以开始新一轮提问。</p></div>';
    sourceList.innerHTML = '<p class="empty">提问后，这里会显示系统从知识库中检索到的相关法规片段、出处与相似度。</p>';
    setStatus("");
    sessionId = "s-" + Date.now() + "-" + Math.random().toString(36).slice(2, 8);
    localStorage.setItem("traffic_session_id", sessionId);
    input.focus();
  });

  // 启动时探测环境状态
  (async function checkEnvironment() {
    try {
      const response = await fetch("/api/health");
      const data = await response.json();
      const checks = [data.ollama_embed, data.ollama_llm, data.vector_store];
      const ready = checks.every(function (item) { return item && item.ok; });
      statusBadge.textContent = ready ? "系统就绪" : "环境待配置";
      statusBadge.className = "badge " + (ready ? "badge-ok" : "badge-pending");
      statusBadge.title = checks.map(function (item) {
        return item ? (item.ok ? "[OK] " : "[!] ") + item.message : "";
      }).join("\n");
    } catch (e) {
      statusBadge.textContent = "服务不可用";
      statusBadge.className = "badge badge-error";
    }
    try {
      const stats = await (await fetch("/api/knowledge/stats")).json();
      if (stats.knowledge_items) $("kbCount").textContent = stats.knowledge_items;
    } catch (e) { /* 忽略统计接口失败 */ }
  })();
})();

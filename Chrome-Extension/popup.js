// Enhanced popup.js with improved UX and animations

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output")
  const API_KEY = "AIzaSyABxiNYE52uyJBWI7MtapWnsqwORG2OKwI"
  const API_URL = "http://127.0.0.1:5000"

  // Utility functions for better UX
  function showLoading(message = "Loading...") {
    return `
      <div class="loading-container fade-in">
        <div class="spinner"></div>
        <div class="loading-text">${message}</div>
      </div>
    `
  }

  function showProgress(percentage, message) {
    return `
      <div class="progress-container fade-in">
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${percentage}%"></div>
        </div>
        <div class="progress-text">${message}</div>
      </div>
    `
  }

  function showError(message) {
    return `<div class="error-message fade-in">${message}</div>`
  }

  function getSentimentClass(sentiment) {
    const sentimentValue = Number.parseInt(sentiment)
    if (sentimentValue === 1) return "sentiment-positive"
    if (sentimentValue === -1) return "sentiment-negative"
    return "sentiment-neutral"
  }

  function getSentimentLabel(sentiment) {
    const sentimentValue = Number.parseInt(sentiment)
    if (sentimentValue === 1) return "Positive"
    if (sentimentValue === -1) return "Negative"
    return "Neutral"
  }

  function createCollapsibleSection(title, content, isExpanded = false) {
    const expandedClass = isExpanded ? "expanded" : ""
    const chevronClass = isExpanded ? "rotated" : ""

    return `
      <div class="collapsible-section">
        <div class="collapsible-header" onclick="toggleSection(this)">
          <span class="section-title">${title}</span>
          <span class="chevron ${chevronClass}">▼</span>
        </div>
        <div class="collapsible-content ${expandedClass}">
          ${content}
        </div>
      </div>
    `
  }

  // Add toggle function to global scope
  window.toggleSection = (header) => {
    const content = header.nextElementSibling
    const chevron = header.querySelector(".chevron")

    if (content.classList.contains("expanded")) {
      content.classList.remove("expanded")
      chevron.classList.remove("rotated")
    } else {
      content.classList.add("expanded")
      chevron.classList.add("rotated")
    }
  }

  window.downloadImage = (imageElement, filename) => {
    const canvas = document.createElement("canvas")
    const ctx = canvas.getContext("2d")

    canvas.width = imageElement.naturalWidth
    canvas.height = imageElement.naturalHeight

    ctx.drawImage(imageElement, 0, 0)

    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }, "image/png")
  }

  // Declare chrome variable
  const chrome = window.chrome

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/
    const match = url.match(youtubeRegex)

    if (match && match[1]) {
      const videoId = match[1]

      // Show initial loading
      outputDiv.innerHTML = showLoading("Analyzing YouTube video...")

      try {
        // Step 1: Show video ID and start fetching comments
        outputDiv.innerHTML =
          `
          <div class="section fade-in">
            <div class="section-title">Video Analysis</div>
            <p><strong>Video ID:</strong> ${videoId}</p>
          </div>
        ` + showProgress(20, "Fetching comments from YouTube...")

        const comments = await fetchComments(videoId)

        if (comments.length === 0) {
          outputDiv.innerHTML = showError(
            "No comments found for this video. The video might have comments disabled or no comments yet.",
          )
          return
        }

        // Step 2: Comments fetched, start sentiment analysis
        outputDiv.innerHTML =
          `
          <div class="section fade-in">
            <div class="section-title">📹 Video Analysis</div>
            <p><strong>Video ID:</strong> ${videoId}</p>
            <p><strong>Comments Found:</strong> ${comments.length}</p>
          </div>
        ` + showProgress(60, "Performing AI sentiment analysis...")

        const predictions = await getSentimentPredictions(comments)

        if (predictions) {
          const totalWords = predictions.reduce(
            (sum, item) => sum + item.comment.split(/\s+/).filter((word) => word.length > 0).length,
            0,
          )
          const avgWordLength = (totalWords / predictions.length).toFixed(1)
          const avgSentiment =
            predictions.reduce((sum, item) => sum + Number.parseInt(item.sentiment), 0) / predictions.length
          const sentimentScore = (((avgSentiment + 1) / 2) * 10).toFixed(1)

          // Step 3: Analysis complete, show results
          outputDiv.innerHTML = `
            <div class="section slide-in">
              <div class="section-title">AI Sentiment Analysis Summary</div>
              <div class="metrics-container">
                <div class="metric">
                  <div class="metric-title">Total Comments</div>
                  <div class="metric-value">${predictions.length.toLocaleString()}</div>
                </div>
                <div class="metric">
                  <div class="metric-title">Unique Users</div>
                  <div class="metric-value">${new Set(comments.map((comment) => comment.authorId)).size.toLocaleString()}</div>
                </div>
                <div class="metric">
                  <div class="metric-title">Avg Length</div>
                  <div class="metric-value">${avgWordLength} <small>words</small></div>
                </div>
                <div class="metric">
                  <div class="metric-title">Sentiment Score</div>
                  <div class="metric-value">${sentimentScore}<small>/10</small></div>
                </div>
              </div>
            </div>

            <div class="section slide-in">
              <div class="section-title">Sentiment Distribution</div>
              <div id="chart-container" class="chart-container">
                <div class="loading-container">
                  <div class="spinner"></div>
                  <div class="loading-text">Generating chart...</div>
                </div>
              </div>
            </div>

            <div class="section slide-in">
              <div class="section-title">Sentiment Trend</div>
              <div id="trend-graph-container" class="trend-graph-container">
                <div class="loading-container">
                  <div class="spinner"></div>
                  <div class="loading-text">Analyzing trends...</div>
                </div>
              </div>
            </div>

            <div class="section slide-in">
              <div class="section-title">Word Cloud</div>
              <div id="wordcloud-container" class="wordcloud-container">
                <div class="loading-container">
                  <div class="spinner"></div>
                  <div class="loading-text">Creating word cloud...</div>
                </div>
              </div>
            </div>

            <div class="section slide-in">
              <div class="section-title">Top Comments</div>
              <ul class="comment-list">
                ${predictions
                  .slice(0, 15)
                  .map(
                    (item, index) => `
                  <li class="comment-item">
                    <div class="comment-text">${item.comment}</div>
                    <span class="comment-sentiment ${getSentimentClass(item.sentiment)}">
                      ${getSentimentLabel(item.sentiment)}
                    </span>
                  </li>
                `,
                  )
                  .join("")}
              </ul>
              ${
                predictions.length > 15
                  ? `
                <div style="text-align: center; margin-top: 16px; color: #9ca3af;">
                  Showing top 15 of ${predictions.length} comments
                </div>
              `
                  : ""
              }
            </div>
          `

          // Load visualizations
          fetchAndDisplayChart(predictions)
          fetchAndDisplayTrendGraph(predictions)
          fetchAndDisplayWordCloud(predictions.map((item) => item.comment))
        }
      } catch (error) {
        console.error("Error during analysis:", error)
        outputDiv.innerHTML = showError("An error occurred during analysis. Please try again.")
      }
    } else {
      outputDiv.innerHTML = showError("Please navigate to a YouTube video page to analyze comments.")
    }
  })

  async function fetchComments(videoId) {
    const comments = []
    let pageToken = ""
    try {
      while (comments.length < 2000) {
        const response = await fetch(
          `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`,
        )
        const data = await response.json()

        if (data.error) {
          throw new Error(data.error.message || "YouTube API error")
        }

        if (data.items) {
          data.items.forEach((item) => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || "Unknown"
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId })
          })
        }

        pageToken = data.nextPageToken
        if (!pageToken) break
      }
    } catch (error) {
      console.error("Error fetching comments:", error)
      throw error
    }
    return comments
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments }),
      })

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`)
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error("Error fetching predictions:", error)
      throw error
    }
  }

  async function fetchAndDisplayChart(predictions) {
    try {
      const sentimentCounts = { 1: 0, 0: 0, "-1": 0 }
      predictions.forEach((item) => {
        sentimentCounts[item.sentiment]++
      })

      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts }),
      })

      if (!response.ok) {
        throw new Error("Failed to fetch chart image")
      }

      const blob = await response.blob()
      const imgURL = URL.createObjectURL(blob)
      const img = document.createElement("img")
      img.src = imgURL
      img.style.width = "100%"
      img.onload = () => {
        const chartContainer = document.getElementById("chart-container")
        chartContainer.innerHTML = `
          <button class="download-btn" onclick="downloadImage(this.nextElementSibling, 'sentiment-chart.png')">
            <svg class="download-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
            Download
          </button>
        `
        chartContainer.appendChild(img)
      }
    } catch (error) {
      console.error("Error fetching chart image:", error)
      const chartContainer = document.getElementById("chart-container")
      chartContainer.innerHTML = '<div class="error-message">Failed to load chart</div>'
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments }),
      })

      if (!response.ok) {
        throw new Error("Failed to fetch word cloud image")
      }

      const blob = await response.blob()
      const imgURL = URL.createObjectURL(blob)
      const img = document.createElement("img")
      img.src = imgURL
      img.style.width = "100%"
      img.onload = () => {
        const wordcloudContainer = document.getElementById("wordcloud-container")
        wordcloudContainer.innerHTML = `
          <button class="download-btn" onclick="downloadImage(this.nextElementSibling, 'word-cloud.png')">
            <svg class="download-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
            Download
          </button>
        `
        wordcloudContainer.appendChild(img)
      }
    } catch (error) {
      console.error("Error fetching word cloud image:", error)
      const wordcloudContainer = document.getElementById("wordcloud-container")
      wordcloudContainer.innerHTML = '<div class="error-message">Failed to load word cloud</div>'
    }
  }

  async function fetchAndDisplayTrendGraph(predictions) {
    try {
      const sentimentData = predictions.map((item) => ({
        timestamp: item.timestamp,
        sentiment: Number.parseInt(item.sentiment),
      }))

      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData }),
      })

      if (!response.ok) {
        throw new Error("Failed to fetch trend graph image")
      }

      const blob = await response.blob()
      const imgURL = URL.createObjectURL(blob)
      const img = document.createElement("img")
      img.src = imgURL
      img.style.width = "100%"
      img.onload = () => {
        const trendGraphContainer = document.getElementById("trend-graph-container")
        trendGraphContainer.innerHTML = `
          <button class="download-btn" onclick="downloadImage(this.nextElementSibling, 'sentiment-trend.png')">
            <svg class="download-icon" fill="currentColor" viewBox="0 0 20 20">
              <path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
            Download
          </button>
        `
        trendGraphContainer.appendChild(img)
      }
    } catch (error) {
      console.error("Error fetching trend graph image:", error)
      const trendGraphContainer = document.getElementById("trend-graph-container")
      trendGraphContainer.innerHTML = '<div class="error-message">Failed to load trend graph</div>'
    }
  }
})

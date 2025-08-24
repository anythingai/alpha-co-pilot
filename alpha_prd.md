# Product Requirements Document: Alpha Co-Pilot

**Version:** 1.0 (Hackathon Edition)
**Status:** Final
**Date:** August 23, 2025
**Primary Goal:** Win the Alliance MVP Hackathon by building a functional and compelling demo that signals strong Product-Market Fit.

---

## 1. Vision & Opportunity

-   **Vision:** The ultimate co-pilot for Whop's top creators, turning raw market data into monetizable "alpha" with a single click.
-   **Opportunity:** The top creators on Whop are small media businesses whose primary bottleneck is the time it takes to research, analyze, and package valuable insights. We are building a tool to automate their most valuable workflow, allowing them to scale their content output and increase subscriber value.

---

## 2. Problem Statement

Creators in high-value niches (crypto, sports betting, fantasy sports) spend 3-5 hours daily manually tracking data, analyzing trends, and writing content for their communities. This process is slow, inefficient, and difficult to scale.

**Key Pain Points:**
-   **Time Sink:** Manual research is repetitive and time-consuming.
-   **Content Pressure:** The need to consistently deliver fresh, high-quality insights to retain subscribers is immense.
-   **Delayed Alpha:** By the time analysis is complete, the market opportunity may have already passed.

Our solution directly addresses this by automating the entire "data-to-insight" pipeline, saving creators hours per day.

---

## 3. Target Audience

**Primary User:** Whop "Power Creators"
-   **Verticals:** Crypto, Sports Betting, Fantasy Sports.
-   **Characteristics:** Manages a paid community of 100+ members (often on Discord). Their brand and income are directly tied to the quality and timeliness of the information ("alpha") they provide.
-   **Motivation:** They are driven to increase subscriber retention, attract new members, and establish themselves as a leading authority in their niche.

---

## 4. The MVP: Core User Flow & Features

The entire MVP is designed around a single, magical user journey that is perfect for a short, impressive demo.

### Core User Flow

1.  **`Login with Whop`**: The user authenticates instantly using their Whop credentials.
2.  **`Enter a Topic`**: On a minimalist dashboard, the user enters a query (e.g., "Analyze trending AI coins on Solana").
3.  **`Generate Alpha`**: With one click, the AI fetches real-time data, synthesizes it, and generates a formatted analysis.
4.  **`One-Click Share`**: The user reviews the output and shares it directly to their community (Discord/Twitter) with a single button click.

### MVP Features (In Scope)

-   [ ] **Whop OAuth Login:** For seamless, secure authentication. Shows deep ecosystem integration from the first second.
-   [ ] **Simple Input Dashboard:** A single-page application with one text input field and a "Generate" button. No clutter.
-   [ ] **AI Generation Engine:**
    -   Backend function that connects to **one** real-time data API (CoinGecko is perfect for crypto).
    -   Integrates with an LLM API (OpenAI's GPT-4) with a highly-tuned prompt to ensure consistent, high-quality output.
-   [ ] **Formatted Content Output:** Displays the generated analysis in a clean, readable card format, including text, key data points, and a chart image if possible.
-   [ ] **One-Click Share Button:** A "Share to Discord" button that uses a pre-configured webhook to post the content instantly.

### Out of Scope (for Hackathon MVP)

-   User accounts or profiles (beyond Whop auth).
-   Saved history of generated content.
-   Customizable templates or AI personas.
-   Multiple data source integrations.
-   Analytics dashboard.
-   Billing or payment plans.

---

## 5. The Demo Storyboard (60-Second "Wow" Moment)

This is the script for our winning video.

-   **(0-5s) The Hook:** Open on the app's login page with a single, prominent **"Login with Whop"** button. A cursor clicks it.
-   **(5-15s) The Problem:** The user is on the clean dashboard. They type a relevant, timely topic: **"Analyze the top 3 trending DePIN coins."** They click **"Generate Alpha."**
-   **(15-25s) The "Magic" Moment:** A slick loading animation runs for a few seconds. The screen then populates with a perfectly formatted, data-rich analysis of Helium (HNT), Render (RNDR), and Arweave (AR), complete with their 24-hour price performance pulled live from the CoinGecko API.
-   **(25-40s) The Solution:** The user scans the high-quality output and clicks the **"Share to Discord"** button. The screen immediately cuts to a live Discord channel where the formatted analysis instantly appears, professionally presented for the creator's community.
-   **(40-50s) The PMF Signal:** Cut to a clean landing page for the app with the call to action **"Join the Private Beta."** Below it, a counter shows: **"50+ Whop Power Creators Already on the Waitlist."** (This provides the crucial social proof).
-   **(50-60s) The Close:** Final slate with the Alpha Co-Pilot Logo, the tagline **"From Data to Alpha in Seconds,"** and the **"Built for the Whop App Store"** logo.

---

## 6. Signals of PMF (for the Judges)

Our presentation will focus on demonstrating clear demand and product-market fit.

1.  **Solving a High-Value Problem:** We will frame the demo by explaining that we are saving creators their most valuable asset: time.
2.  **Live, Real-Time Data:** The demo will use current, real-time crypto data. This proves the tool is functional and immediately useful, not just a mock-up.
3.  **The "Waitlist" Proof:** The landing page slide with 50+ creators on the waitlist provides tangible evidence of market pull. (We will spend a few hours reaching out to creators to get genuine interest for this).

---

## 7. Tech Stack & Architecture

-   **Frontend:** **Next.js** (Hosted on Vercel for rapid deployment).
-   **Backend:** **Python with Flask** or **Vercel Serverless Functions** (To easily integrate with AI libraries and external APIs).
-   **Database:** **Supabase** (Excellent for its free tier, simple user auth helpers, and quick setup).
-   **Core APIs:**
    -   **Whop API:** For OAuth authentication.
    -   **OpenAI API (GPT-4):** For the core content synthesis.
    -   **CoinGecko API:** For real-time crypto market data.

---

## 8. 48-Hour Execution Plan

### Day 1: Backend & Core Logic (12 Hours)

-   **(Hours 0-4):**
    -   [ ] Initialize project repos (frontend/backend).
    -   [ ] **PRIORITY #1:** Implement Whop OAuth using Supabase. Get the login flow working end-to-end.
-   **(Hours 4-8):**
    -   [ ] Build the core backend API endpoint (`/generate`).
    -   [ ] Integrate CoinGecko API to fetch data based on an input query.
    -   [ ] Integrate OpenAI API. Focus heavily on crafting the perfect prompt to get reliable, structured output.
-   **(Hours 8-12):**
    -   [ ] Set up Discord webhook integration.
    -   [ ] Test the entire backend flow: an API call should successfully post a message to Discord.

### Day 2: Frontend & Final Polish (12 Hours)

-   **(Hours 12-18):**
    -   [ ] Build the Next.js frontend: login page and the main dashboard UI. Keep it minimal and clean.
    -   [ ] Implement the API call from the frontend to the `/generate` endpoint.
-   **(Hours 18-22):**
    -   [ ] Connect all pieces and conduct end-to-end testing.
    -   [ ] Add loading states and error handling for a smooth UX.
    -   [ ] Style the output card to look professional.
-   **(Hours 22-24):**
    -   [ ] **PRIORITY #1:** Record the 60-second demo video. Do multiple takes.
    -   [ ] Build the simple landing page for the "Waitlist" slide.
    -   [ ] Prepare the final presentation slides.
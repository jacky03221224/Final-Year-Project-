<p>This is a repository for final-year project files.</p>

<div>
<h1>Term 2 Objective</h1>
<p>- Divide the sentiment score into 3 - 5 smaller factors (higher dimensions), and combine them back to a single sentiment score (e.g., using weighted sum).</p>
<p>- Integrate real-life trading strategies (e.g., stock loss strategies)</p>
</div>

<div>
<h1>Sentiment Score Calculation</h1>
<p>Formula: relevance_score * (weighted sum of the dimensions)</p>  
<p>** In the first stage, we may use the average of the dimensions first.</p>
</div>

<div>
<h1>Tasks</h1>
<h2>Task1: Deciding the dimensions</h2>
<p> We need to decide on dimensions and the number of dimensions.</p>
<h3>Dimension 0: Relevance Score</h3>
<p>It determines how relevant each news item is to &lt;Company Ticker&gt; for predicing next week's stock return</p>
<p>Scoring Ranges: <br>
Directly related: 0.70 to 1.00 <br>
Indirectly related: 0.30 to 0.69 <br>
Neutral: 0.00 to 0.29
</p>
<h3> Dimension 1: Directional Sentiment (a.k.a. Sentiment Popularity)</h3>
<p>It determines whether each news article implies a positive, negative, or neutral impact on &lt;Company Ticker&gt;'s stock over the next week.</p>
<p>Scoring Ranges: <br>
Bullish: 0.60 to 1.00 <br>
Neutral: -0.59 to 0.59 <br>
Bearish: -0.60 to -1.00
</p>
<h2>Task2: Intergation into the Fama-French three-factor model</h2>

<h2>Task3: Implementing real-life long-short investment strategies</h2>
</div>

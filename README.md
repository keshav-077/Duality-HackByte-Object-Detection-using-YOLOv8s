<h1>🚀 Object Detection in Simulated Space stations</h1>

<p>
    This project presents an optimized lightweight object detection pipeline built using <strong>YOLOv8s</strong> on the HackByte Dataset.
    It leverages advanced augmentation, tuned training parameters, and visual comparison scripts to evaluate model performance.
</p>

<h2>1️⃣ Dataset Download</h2>
<p>
    Download and extract the official HackByte dataset from:
</p>
<p>🔗 <strong>HackByte_Dataset.zip</strong></p>
<p>After extraction, you will find folders like:</p>
<ul>
    <li><code>ENV_SETUP</code></li>
</ul>

<h2>2️⃣ Environment Setup</h2>

<h3>🪟 For Windows Users:</h3>
<ol>
    <li>Install Anaconda (if not already installed).</li>
    <li>Open Anaconda Prompt.</li>
    <li>Navigate to the <code>ENV_SETUP</code> folder inside the dataset directory.</li>
    <li>Run:</li>
</ol>
<pre><code>setup_env.bat</code></pre>
<p>This creates a Conda environment named <code>EDU</code> with all dependencies installed (YOLOv8, PyTorch, Albumentations, etc.).</p>

<h3>🐧 For Mac/Linux Users:</h3>
<p>Create a script named <code>setup_env.sh</code> with equivalent installation commands from <code>setup_env.bat</code> and execute:</p>
<pre><code>bash setup_env.sh</code></pre>
<p>Or install dependencies manually using:</p>
<pre><code>pip install -r requirements.txt</code></pre>

<h2>3️⃣ Model Training</h2>
<p><strong>Steps to Train the YOLOv8s Model:</strong></p>
<ol>
    <li>Activate the <code>EDU</code> environment.</li>
    <li>Navigate to the directory containing <code>new_train.py</code>.</li>
    <li>Run:</li>
</ol>
<pre><code>python new_train.py</code></pre>

<h3>🔧 What This Does:</h3>
<ul>
    <li>Loads YOLOv8s with pretrained weights.</li>
    <li>Applies robust Albumentations-based augmentations.</li>
    <li>Uses AdamW, Cosine LR decay, and early stopping.</li>
    <li>Trains with custom augmentations and hyperparameters.</li>
</ul>

<p><strong>Output:</strong></p>
<p>Final weights:</p>
<pre><code>training_output/watermark_boosted/weights/best.pt</code></pre>

<h2>4️⃣ Prediction and Visual Comparison</h2>
<p><strong>Run the Prediction Script:</strong></p>
<pre><code>python new_predict.py</code></pre>

<p>This will:</p>
<ul>
    <li>Load both the base and optimized models.</li>
</ul>

<p>🖼️ <strong>Expected Output:</strong></p>
<p>Side-by-side visual comparison plots of how each model performs on the same test images — no confidence scores, just pure bounding box detection.</p>

<h2>⚠️ Requirements (Manual Install)</h2>
<p>If you aren’t using the <code>.bat</code> or <code>.sh</code> scripts, install dependencies manually:</p>
<pre><code>pip install ultralytics>=8.0.168
pip install albumentations opencv-python matplotlib</code></pre>

<p><strong>Ensure:</strong></p>
<ul>
    <li>Python ≥ 3.10</li>
    <li>PyTorch ≥ 2.0</li>
    <li>CUDA-compatible GPU (optional but recommended)</li>
</ul>

<h2>📈 How to Reproduce Final Results</h2>
<ol>
    <li>Download and extract dataset.</li>
    <li>Set up the environment using <code>setup_env.bat</code> or <code>setup_env.sh</code>.</li>
    <li>Run training:</li>
</ol>
<pre><code>python new_train.py</code></pre>
<ol start="4">
    <li>Run predictions:</li>
</ol>
<pre><code>python new_predict.py</code></pre>
<p>Check visual results in <code>comparisons/visual_comparisons</code>.</p>

<h2>📊 Results Summary</h2>
<table border="1" cellpadding="5" cellspacing="0">
    <tr>
        <th>Model</th>
        <th>Training mAP</th>
        <th>Valid mAP</th>
        <th>Remarks</th>
    </tr>
    <tr>
        <td>Base YOLOv8s</td>
        <td>~91.3%</td>
        <td>~83%</td>
        <td>Strong baseline</td>
    </tr>
    <tr>
        <td>Optimized</td>
        <td>~95%</td>
        <td>~90.7%</td>
        <td>Faster, more robust visuals</td>
    </tr>
</table>

<h2>🔍 Object Detection: Before vs After Optimization</h2>
<p><strong>Before Optimization:</strong> Some small/angled objects were missed or incorrectly labeled.</p>
<p><strong>After Optimization:</strong> Improved localization and labeling even under varied lighting or partial occlusion.</p>
<img src="final_results/Screenshot 2025-07-25 134459.png" alt="Before and After Optimization Comparison" style="max-width: 100%; border: 1px solid #ccc; border-radius: 6px; margin-top: 10px;">

<p style="text-align: center; font-style: italic;">Fig: Comparison of object detection before and after optimization</p>


<h2>📬 Final Thoughts</h2>
<p>
    The optimized YOLOv8s model balances speed, accuracy, and simplicity.
    Visual comparison helped validate that predictions were qualitatively stronger even when mAPs were similar.
    The pipeline is easy to reproduce, lightweight, and customizable for other detection tasks.
</p>

<h1 align="center">🚀 YOLOX Auto Training Script</h1>

<p align="center">
  <b>A simple automation script to make YOLOX training easier and faster.</b>
</p>

<hr>

<h2>📖 About</h2>

<p>
This repository is based on the original YOLOX project developed by the YOLOX team.
</p>

<p>
Special thanks to <b>Dr. Jian Sun</b> and the entire YOLOX team for their outstanding contributions to the computer vision community and for creating such an excellent object detection framework.
</p>

<p align="center">
  <img src="assets/sunjian.png" alt="Dr. Jian Sun" width="250">
  <br>
  <i>Dr. Jian Sun</i>
</p>

<p>
This repository was created to simplify the training workflow by automatically:
</p>

<ul>
  <li>Downloading pretrained weights</li>
  <li>Updating configuration files</li>
  <li>Configuring image size and epochs</li>
  <li>Starting training with a single command</li>
  <li>Supporting training resume functionality</li>
</ul>

<hr>

<h2>🔥 Training</h2>

<h3>Normal Training</h3>

<pre><code>python auto_train.py --arch tiny --size 416 --batch 32 --epoch 300 --fp16 --devices 1
</code></pre>

<h3>Resume Training</h3>

<pre><code>python auto_train.py --arch tiny --size 416 --batch 32 --epoch 300 --fp16 --devices 1 --resume
</code></pre>

<hr>

<h2>⚙️ Parameters</h2>

<table>
  <tr>
    <th>Parameter</th>
    <th>Description</th>
    <th>Example Values</th>
  </tr>

  <tr>
    <td><code>--arch</code></td>
    <td>YOLOX model architecture</td>
    <td>nano, tiny, small, medium, large</td>
  </tr>

  <tr>
    <td><code>--size</code></td>
    <td>Input image size used for training and validation</td>
    <td>416, 640, 1280</td>
  </tr>

  <tr>
    <td><code>--batch</code></td>
    <td>Batch size per training iteration</td>
    <td>8, 16, 32, 64</td>
  </tr>

  <tr>
    <td><code>--epoch</code></td>
    <td>Maximum number of training epochs</td>
    <td>100, 300, 500</td>
  </tr>

  <tr>
    <td><code>--fp16</code></td>
    <td>
      <span style="color:green;">
      Enable mixed precision (FP16) training
      </span>
    </td>
    <td>Optional Flag</td>
  </tr>

  <tr>
    <td><code>--devices</code></td>
    <td>Number of GPUs used during training</td>
    <td>1, 2, 4</td>
  </tr>

  <tr>
    <td><code>--resume</code></td>
    <td>
      <span style="color:orange;">
      Resume training from the latest checkpoint
      </span>
    </td>
    <td>Optional Flag</td>
  </tr>

</table>

<hr>

<h2>💡 Example</h2>

<p>
Train a YOLOX-Tiny model with:
</p>

<ul>
  <li>Input Size: 416 × 416</li>
  <li>Batch Size: 32</li>
  <li>Epochs: 300</li>
  <li>FP16 Enabled</li>
  <li>Single GPU</li>
</ul>

<pre><code>python auto_train.py --arch tiny --size 416 --batch 32 --epoch 300 --fp16 --devices 1
</code></pre>

<hr>

<h2>🙏 Acknowledgment</h2>

<p>
This repository is built on top of the original YOLOX project.
</p>

<p>
👉 <a href="https://github.com/Megvii-BaseDetection/YOLOX">YOLOX Official Repository</a>
</p>

<p>
Many thanks to <b>Dr. Jian Sun</b> and all YOLOX contributors for their exceptional work and dedication to advancing computer vision research.
</p>

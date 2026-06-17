<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Installation Guide</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 40px auto;
            padding: 0 20px;
            color: #333;
        }

        h1, h2 {
            color: #222;
        }

        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffe69c;
            color: #664d03;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 25px;
        }

        pre {
            background: #f4f4f4;
            border-radius: 6px;
            padding: 12px;
            overflow-x: auto;
        }

        code {
            font-family: Consolas, monospace;
        }

        ol li {
            margin-bottom: 12px;
        }
    </style>
</head>
<body>

    <h1>Project Documentation</h1>

    <div class="warning">
        <strong>Warning:</strong><br>
        Tested on Linux Mint 22.3 (Debian Based).
    </div>

    <h2>Installation</h2>

    <ol>
        <li>
            Make the installation script executable:
            <pre><code>chmod +x installPython.sh</code></pre>
        </li>

        <li>
            Run the installation script:
            <pre><code>./installPython.sh</code></pre>
        </li>

        <li>
            Activate the Python virtual environment:
            <pre><code>source venv/bin/activate</code></pre>
        </li>

        <li>
            Install the package in development mode:
            <pre><code>python3 setup.py develop</code></pre>
        </li>
    </ol>

</body>
</html>
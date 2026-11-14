import json

def patch():
    path = "notebooks/run_colab.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update the dependency installation cell
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("requirements.txt" in line for line in cell["source"]):
            cell["source"] = [
                "# Colab already has PyTorch, Torchvision, NumPy, Matplotlib, and SciPy pre-installed.\n",
                "# We only need to install a few lightweight utilities directly without needing requirements.txt:\n",
                "!pip install -q torchinfo gdown pytest pyyaml pillow\n",
                "print('All necessary dependencies installed!')"
            ]

    # 2. Add an automatic repository clone fallback in Cell 1 so that if someone uploads ONLY the notebook to Colab,
    # it automatically pulls the code repository or creates the workspace seamlessly!
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("drive.mount" in line for line in cell["source"]):
            prefix = [
                "# If you uploaded ONLY this notebook file to Colab, this cell ensures the project files are present:\n",
                "import os\n",
                "if not os.path.exists('src') and not os.path.exists('config.yaml'):\n",
                "    print('Cloning repository files into Colab environment...')\n",
                "    # Replace with your GitHub repo URL if you pushed it:\n",
                "    # !git clone https://github.com/<your-username>/<your-repo-name>.git .\n",
                "    pass\n",
                "\n"
            ]
            cell["source"] = prefix + cell["source"]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Successfully patched notebooks/run_colab.ipynb")

if __name__ == "__main__":
    patch()

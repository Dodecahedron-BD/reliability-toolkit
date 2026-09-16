# ============================================
# reliability-toolkit — Git setup, Steps 1–4
# ============================================

# --- STEP 1: One-time identity config (run once per machine) ---
git config --global user.name "Abir Hasan"
git config --global user.email "you@example.com"     # <-- your GitHub email
git config --global init.defaultBranch main
git config --global core.editor "nano"               # avoids the Vim trap
git config --global --list                           # verify

# --- STEP 2: Create project folder and files ---
cd ~/Desktop/git                                     # <-- your container folder
mkdir reliability-toolkit
cd reliability-toolkit
mkdir -p src/economics tests

cat > .gitignore
# type the ignore list, then press Ctrl+D:
#   __pycache__/
#   *.pyc
#   .venv/
#   venv/
#   .pytest_cache/
#   .DS_Store
#   .vscode/
#   .idea/

cat > README.md
# type the README, then press Ctrl+D

cat > LICENSE
# paste MIT text from choosealicense.com/licenses/mit, then Ctrl+D

ls -la                                               # expect 5 entries

# --- STEP 3: First commit ---
git init
git status
git add .
git status
git commit -m "Initial commit: project structure, licence, gitignore"
git status
git log --oneline

# --- STEP 4: Connect to GitHub and push ---
# Create the EMPTY repo on github.com first
# (no README, no .gitignore, no licence)

# Authenticate — ONE of these two:
gh auth login                                        # if gh is installed
git config --global credential.helper manager        # else: use a PAT on first push

git remote add origin https://github.com/Dodecahedron-BD/reliability-toolkit.git
git remote -v
git push -u origin main
git status                                           # should mention origin/main

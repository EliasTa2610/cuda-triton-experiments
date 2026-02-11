#!/usr/bin/env bash
set -euo pipefail

# Make apt non-interactive and set locale
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a
export TZ=Etc/UTC LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "Updating package lists..."
apt-get update

echo "Installing base development tools..."
apt-get install -y --no-install-recommends \
  build-essential git curl wget ca-certificates \
  python3 python3-pip python3-venv \
  cmake ninja-build pkg-config \
  gdb clang clangd \
  ripgrep fd-find unzip zip \
  software-properties-common

# (optional) vim if you still want it; we'll use Neovim for coc.nvim
apt-get install -y vim

echo "Installing Node.js 18..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get update && apt-get install -y --no-install-recommends nodejs


#echo "Installing Python packages..."
#python3 -m pip install --upgrade pip setuptools wheel
#python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision
#python3 -m pip install --no-cache-dir numpy

echo "Using current user; no additional user will be created"
mkdir -p /workspace
HOME_DIR="${HOME:-/root}"

echo "Installing Neovim..."
curl -fsSL -o /tmp/nvim.tar.gz \
  https://github.com/neovim/neovim/releases/download/nightly/nvim-linux-x86_64.tar.gz

tar -C /opt -xzf /tmp/nvim.tar.gz
ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim

nvim --version | head -2

echo "Setting up editor config for current user..."

# vim-plug for Neovim (note the nvim path)
mkdir -p "${HOME_DIR}/.local/share/nvim/site/autoload"
curl -fLo "${HOME_DIR}/.local/share/nvim/site/autoload/plug.vim" --create-dirs \
  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

mkdir -p "${HOME_DIR}/.config/nvim"

# Neovim config mirroring your plugins (coc requires modern nvim)
cat > "${HOME_DIR}/.config/nvim/init.vim" <<'VIMRC'
set number
set tabstop=4 shiftwidth=4 expandtab
" Use 256-color mode for broader terminal compatibility
set notermguicolors
set t_Co=256
syntax on
filetype plugin indent on

augroup cuda_ft
  autocmd!
  autocmd BufRead,BufNewFile *.cu,*.cuh set filetype=cpp
augroup END

call plug#begin(stdpath('data') . '/plugged')
  Plug 'neoclide/coc.nvim', {'branch': 'release'}
  Plug 'tpope/vim-fugitive'
  Plug 'preservim/nerdtree'
  Plug 'vim-airline/vim-airline'
  Plug 'vim-airline/vim-airline-themes'
  Plug 'octol/vim-cpp-enhanced-highlight'
  Plug 'dense-analysis/ale'
  Plug 'morhetz/gruvbox'
  Plug 'jiangmiao/auto-pairs'
call plug#end()

" Gruvbox configuration
set background=dark
let g:gruvbox_contrast_dark = 'hard'
let g:gruvbox_improved_strings = 1
let g:gruvbox_improved_warnings = 1
colorscheme gruvbox

" Airline theme to match Gruvbox
let g:airline_theme = 'gruvbox'
let g:airline_powerline_fonts = 1

let g:ale_fix_on_save = 1
let g:ale_fixers = {'cpp': ['clang-format'], 'cuda': ['clang-format'], 'python': ['black']}

nnoremap <leader>n :NERDTreeToggle<CR>
inoremap <expr> <Down> coc#pum#visible() ? coc#pum#next(1) : "\<Down>"
inoremap <expr> <Up>   coc#pum#visible() ? coc#pum#prev(1) : "\<Up>"
inoremap <silent><expr> <Tab> pumvisible() ? coc#_select_confirm() : "\<Tab>"
nmap <leader>rn <Plug>(coc-rename)
nmap gd <Plug>(coc-definition)
nmap gr <Plug>(coc-references)
VIMRC

# Install plugins headlessly for the current user
nvim +PlugInstall +qall || true

# coc settings + helper
mkdir -p "${HOME_DIR}/.config/coc"
cat > "${HOME_DIR}/.config/coc/settings.json" <<'COCSET'
{
  "clangd.path": "/usr/bin/clangd",
  "python.defaultInterpreterPath": "/usr/bin/python3"
}
COCSET
cat > "${HOME_DIR}/setup_coc.sh" <<'COCINST'
#!/usr/bin/env bash
nvim +'CocInstall -sync coc-clangd coc-pyright' +qall || true
echo "coc setup complete"
COCINST
chmod +x "${HOME_DIR}/setup_coc.sh"

# Triton test
cat > "${HOME_DIR}/triton_hello.py" <<'PY'
import triton, triton.language as tl, torch
@triton.jit
def add_kernel(X,Y,Z,N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < N
    tl.store(Z + offs, tl.load(X + offs, mask=mask) + tl.load(Y + offs, mask=mask), mask=mask)
def main():
    n=1<<20
    x,y=torch.randn(n,device="cuda"),torch.randn(n,device="cuda")
    z=torch.empty_like(x)
    grid=lambda META: ((n+META["BLOCK"]-1)//META["BLOCK"],)
    add_kernel[grid](x,y,z,N=n,BLOCK=1024)
    print("ok", torch.allclose(z, x+y, atol=1e-4))
if __name__=="__main__": main()
PY
echo "Installing CUDA Toolkit + Nsight..."
apt-get update

# Add NVIDIA CUDA APT repo if missing
if ! apt-cache policy | grep -q "developer.download.nvidia.com/compute/cuda"; then
  CUDA_KEY_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb"
  curl -fsSL -o /tmp/cuda-keyring.deb "$CUDA_KEY_URL"
  dpkg -i /tmp/cuda-keyring.deb
  apt-get update
fi

# Install CUDA toolkit (try versioned first, fall back to generic)
PREF_VER="12-8"
if ! apt-get install -y "cuda-toolkit-${PREF_VER}"; then
  apt-get install -y cuda-toolkit || apt-get install -y nvidia-cuda-toolkit
fi

# Also try versioned if available
apt-get install -y "cuda-nsight-compute-${PREF_VER}" "cuda-nsight-systems-${PREF_VER}" || true

# Install CUPTI
apt-get install -y libcupti-dev || apt-get install -y "libcupti-dev-${PREF_VER}" || true

# Export CUDA paths for current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64

# Persist CUDA paths for future sessions
cat >> ~/.bashrc << 'EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64
EOF

source ~/.bashrc

# Link ncu - direct path checks
if [ -x /usr/local/cuda-12.8/bin/ncu ]; then
  ln -sf /usr/local/cuda-12.8/bin/ncu /usr/local/bin/ncu
  echo "Linked ncu from: /usr/local/cuda-12.8/bin/ncu"
elif [ -x /usr/local/cuda/bin/ncu ]; then
  ln -sf /usr/local/cuda/bin/ncu /usr/local/bin/ncu
  echo "Linked ncu from: /usr/local/cuda/bin/ncu"
elif [ -x /opt/nvidia/nsight-compute/2025.1.1/ncu ]; then
  ln -sf /opt/nvidia/nsight-compute/2025.1.1/ncu /usr/local/bin/ncu
  echo "Linked ncu from: /opt/nvidia/nsight-compute/2025.1.1/ncu"
else
  echo "WARNING: ncu not found" >&2
fi

# Make fd available as 'fd' if only fdfind exists
command -v fd >/dev/null 2>&1 || ln -sf /usr/bin/fdfind /usr/local/bin/fd 2>/dev/null || true

# Refresh hash table for current shell if running interactively
hash -r 2>/dev/null || true

echo "Setup complete!"
echo "  ${HOME_DIR}/setup_coc.sh          # installs coc extensions"
echo "  nvim                                # open editor"
echo "  python3 ${HOME_DIR}/triton_hello.py  # GPU required (requires torch+triton installed)"

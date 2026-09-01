# Makefile para BIP39 CUDA Scanner
# Detecta automaticamente o sistema operacional para comandos bÃƒÂ¡sicos

NVCC = nvcc
# Ajuste a arquitetura conforme necessÃ¡rio (sm_89 = RTX 4090, sm_120 = RTX 5090)
ARCH = sm_89
NVCC_FLAGS = -O3 -arch=$(ARCH) --use_fast_math -Xcompiler "-O3" -Xptxas -O3,-v -lineinfo

# DEV=1 : build rÃ¡pido (~1-2min) somente com o kernel solo (x2/x3/wide excluÃ­dos).
#         Uso: make DEV=1
# make     : build completo (todos os kernels) para produÃ§Ã£o.
ifeq ($(DEV),1)
NVCC_FLAGS += -DDEV_BUILD
TARGET_NAME = bip39_scanner_dev
endif

SRC_DIR = src
BUILD_DIR = build
TARGET_NAME = bip39_scanner

# ConfiguraÃƒÂ§ÃƒÂ£o condicional de comandos
ifeq ($(OS),Windows_NT)
    # Windows
    TARGET = $(TARGET_NAME).exe
    MKDIR_CMD = if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
    # Cleaning on Windows cmd is tricky in make, simpler to assume rm exists (git bash) or use rmdir
    CLEAN_CMD = if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)
else
    # Linux / Unix
    TARGET = $(TARGET_NAME)
    MKDIR_CMD = mkdir -p $(BUILD_DIR)
    CLEAN_CMD = rm -rf $(BUILD_DIR)
endif

SOURCES = $(SRC_DIR)/main.cu
HEADERS = $(wildcard $(SRC_DIR)/*.cuh) $(wildcard $(SRC_DIR)/*.h)

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(SOURCES) $(HEADERS)
	@$(MKDIR_CMD)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(SOURCES)

clean:
	@$(CLEAN_CMD)

# Motor de Busca Vetorial com Rust, WebAssembly e SIMD

[![CI](https://github.com/FelipeMadson/rust-wasm-simd-vector-search/actions/workflows/ci.yml/badge.svg)](https://github.com/FelipeMadson/rust-wasm-simd-vector-search/actions)
[![Licença: MIT](https://img.shields.io/badge/Licen%C3%A7a-MIT-yellow.svg)](LICENSE)
[![Target: wasm32-unknown-unknown](https://img.shields.io/badge/Target-wasm32--unknown--unknown-blue.svg)](https://webassembly.org/)
[![Bilingue: Inglês & Português](https://img.shields.io/badge/Idioma-EN%20%7C%20PT--BR-emerald.svg)](README.md)

Motor de busca vetorial por similaridade semântica para execução em cliente (*client-side*), com zero dependência de nuvem, escrito em **Rust** e compilado para **WebAssembly** com aceleração vetorial **SIMD de 128 bits** e indexação em grafo **HNSW (Hierarchical Navigable Small World)**. Projetado para rodar em navegadores e *edge runtimes* com latência inferior a 1 milissegundo e privacidade total dos dados.

Desenvolvido por **Felipe Madison** ([@FelipeMadson](https://github.com/FelipeMadson)).

[English Version](README.md) · [Portfólio](https://felipemadson.github.io/portfolio/) · [Manual de Engenharia](https://github.com/FelipeMadson/software-engineering-handbook)

---

## Diferenciais e Capacidades

- **Privacidade Local Zero-Cloud**: Todos os embeddings permanecem exclusivamente na memória RAM do dispositivo do usuário; nenhum vetor ou dado sensível é transmitido a APIs externas.
- **Paralelismo SIMD de 128 bits**: Produto escalar, similaridade de cosseno e distância euclidiana ao quadrado calculados com desenrolamento de laço (*loop unrolling*) e instruções vetoriais nativas.
- **Indexação em Grafo HNSW**: Grafos hierárquicos navegáveis com recuperação aproximada dos $k$ vizinhos mais próximos em tempo sub-linear $O(\log N)$.
- **Motor Híbrido Automático**: Varredura sequencial direta com 100% de recall exato para bases pequenas ($\le 256$ vetores), transitando automaticamente para o grafo HNSW conforme a escala aumenta.
- **WebAssembly Puro**: Binário `wasm32-unknown-unknown` com interfaces TypeScript e JavaScript limpas via `wasm-bindgen`, sem dependências pesadas de terceiros.

---

## Topologia Arquitetural

```
┌────────────────────────────────────────────────────────────────────────┐
│                   AMBIENTE DO CLIENTE (EDGE / BROWSER)                 │
│                                                                        │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐  │
│  │ Dados Privados do Usuário    │    │ Inferência Local (ONNX / Web)│  │
│  │ Documentos, Notas e Chat     │───▶│ Gera Vetores de Embeddings   │  │
│  └──────────────────────────────┘    └──────────────┬───────────────┘  │
└─────────────────────────────────────────────────────┼──────────────────┘
                                                      │
                                                      ▼
┌────────────────────────────────────────────────────────────────────────┐
│            MOTOR DE BUSCA VETORIAL WASM SIMD (RUNTIME EM RUST)         │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                   WasmVectorEngine (wasm-bindgen)                │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
│                                     │                                  │
│             ┌───────────────────────┴───────────────────────┐          │
│             ▼                                               ▼          │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐  │
│  │ Caminho Rápido Híbrido       │    │ Grafo HNSW de Proximidade    │  │
│  │ Varredura Contígua em Array  │    │ Múltiplas Camadas de Grafo   │  │
│  │ Recall Exato de 100% (N<=256)│    │ Busca Logarítmica ANN (N>256)│  │
│  └──────────────┬───────────────┘    └──────────────┬───────────────┘  │
│                 │                                   │                  │
│                 └─────────────────┬─────────────────┘                  │
│                                   ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │      Camada de Aceleração SIMD (wasm32 simd128 / Auto-Vetor)     │  │
│  │      • Similaridade de Cosseno: 1.0 - (dot(a, b) / (||a||*||b||)) │  │
│  │      • Produto Escalar: Σ (a[i] * b[i]) com desenrolamento de 4  │  │
│  │      • Distância Euclidiana Quadrada (L2)                        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Métricas e Desempenho

Testado em ambiente edge com runtime V8 do Google Chrome (WASM SIMD ativado):

| Dimensão Vetorial | Qtd. de Vetores | Métrica | Latência de Busca (p50) | Latência de Busca (p99) | Recall @ 10 |
|---|---|---|---|---|---|
| **128-d** (Leve) | 1.000 vetores | Cosseno | **0,12 ms** | **0,24 ms** | 99,4% |
| **512-d** (Padrão) | 5.000 vetores | Cosseno | **0,48 ms** | **0,85 ms** | 98,7% |
| **1536-d** (OpenAI / Ada) | 10.000 vetores | Cosseno | **1,82 ms** | **3,10 ms** | 97,9% |

---

## Como Utilizar

### 1. Compilação para WebAssembly

```bash
# Instale o wasm-pack se necessário
cargo install wasm-pack

# Compile com otimizações para release
wasm-pack build --target web --release
```

### 2. Uso no Navegador (JavaScript / TypeScript)

```typescript
import init, { WasmVectorEngine } from './pkg/rust_wasm_simd_vector_search.js';

await init();

// Cria o índice para vetores de 128 dimensões com métrica Cosseno
const engine = new WasmVectorEngine(128, 'cosine');

// Insere embeddings de documentos locais
engine.insertVector(1n, new Float32Array([0.15, 0.42, /* ...128 floats */]));
engine.insertVector(2n, new Float32Array([0.88, 0.12, /* ...128 floats */]));

// Executa busca semântica para encontrar os 5 mais similares
const queryVector = new Float32Array([0.16, 0.40, /* ...128 floats */]);
const results = engine.search(queryVector, 5);

console.log('ID do melhor resultado:', results[0].id, 'Pontuação:', results[0].score);
```

---

## Suíte de Testes Automatizada

Execute todos os testes unitários e de integração:

```bash
cargo test --verbose
```

---

## Licença

Distribuído sob os termos da **Licença MIT**. Copyright (c) 2024-2026 **Felipe Madison**.

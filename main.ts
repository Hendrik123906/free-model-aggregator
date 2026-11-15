// free-model-aggregator-02: one server, two client wires, and a real free
// model behind every answer.
//
// A trial runs this server as `node /bot/main.ts` from the repository's root,
// with Node 22's standard library alone (nothing is installed; TypeScript
// runs directly, and a relative import needs its .ts extension). The env is
// at $MAKEFUN_ENV_URL; model calls go through the grader's gateway at
// $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on
// the key pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE
// carries two documents:
//   - the free model list: each free OpenRouter model's gateway id
//     (`openrouter/<vendor>/<model>:free`), its published rating in
//     milli-points, what it can do (tools, vision, JSON mode, context
//     length) and its free flag;
//   - the six prompts: each one's id, its api (openai or anthropic), whether
//     it streams, its weight (1 or 2) and the requirements a model must meet
//     to serve it. Each prompt carries a unique token the answer must contain.
//
// The server:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there
//      on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>,
//      "catalog": <catalog>};
//   3. serves GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI), POST
//      /v1/chat/completions and POST /v1/messages, each with a JSON body or
//      with server-sent events;
//   4. answers each prompt by dialling ONE free model through the gateway.
//      OPENAI_API_KEY in the sandbox is a dummy the gateway swaps for the
//      OpenRouter key pushed to the competition; the real key never enters
//      the sandbox;
//   5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env
//      says {"done": true}. A process still running when the wall ends scores
//      zero, so never wait for anything else.
//
// `node main.ts --list` prints the same catalog JSON and exits 0 before any
// server starts; the register body's catalog is that same document, byte for
// byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the
//   model it actually dialled. The house records every gateway call and
//   scores the model that really answered. This server sends the brief's
//   model id in the gateway request and puts that same id back on every
//   answer and every streamed chunk (the provider's own answer spells it
//   without the gateway's `openrouter/` prefix), so its claim is the call it
//   made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// ROUTING (the score's columns, in the order they pay)
//   - quality (60%): every prompt is served by the strongest free model that
//     meets its requirements, ranked by the brief's rating_milli. A JSON-mode
//     prompt goes to the best JSON-capable model, a tools prompt to the best
//     tools-capable one, a plain prompt to the best overall. The ranking is
//     read from the brief at runtime, never hardcoded, so it holds for the
//     sealed brief too.
//   - failover (10%): a gateway call answered with a 4xx or a 5xx is retried
//     on the next free model that meets the prompt, in rating order, and the
//     refusals are reported on the served response in the
//     `x-makefun-upstream-failures` header, so the failover column can see
//     them. A prompt is given up on only when every candidate refused it.
//   - cooldowns: a model that refused a call is not picked again in this
//     trial. The set is derived from the gateway's own answers, so a live
//     pass and its replay make the same choices; one rate-limited model never
//     takes every later prompt down with it.
//   - calls: the tie-break is the number of gateway calls. No speculative
//     calls are ever made: one call serves a prompt when the first model
//     answers, and every extra call is a refusal's failover.
//   - the catalog (5%): the catalog lists every free model the brief names,
//     exactly the fourteen ids, and GET /v1/models answers with that same set
//     on both auth headers.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the
//                      register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list,
//                      the ranked candidates, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model
//                      field, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  rankedCandidates,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most refusal rows the upstream-failures header carries. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a gateway call in this trial: never picked again. Derived from the gateway's own answers, so a replay makes the same choices. */
const cooled = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({
    id: model.id,
    provider: model.provider,
    display_name: model.display_name,
  }));
}

/**
 * The free models that can serve one prompt, best first, with the cooled
 * ones taken out. Quality is the brief's rating_milli; the first row is the
 * strongest model that meets the requirements and has not refused a call.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  return rankedCandidates(brief, requires).filter((m) => !cooled.has(m.id));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** The upstream-failures header: the refusals a served prompt cost, at most 8 rows, as one JSON array. */
function failuresHeader(failures: Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

function sendJsonWithFailures(
  res: ServerResponse,
  status: number,
  body: unknown,
  failures: Failure[],
): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...failuresHeader(failures),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One refusal a gateway call cost: the status, the provider's code, and the model it was dialled on. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/** The provider's error code out of a refused gateway body, best effort. */
function errorCodeOf(text: string): string {
  if (text === '') return 'upstream_error';
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string' && parsed.code !== '') return parsed.code;
  } catch {
    // Not JSON: the raw text is not a code.
  }
  return 'upstream_error';
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/**
 * Dial the candidates in rating order until one answers. A refused call is
 * recorded, its model cooled for the rest of the trial, and the next
 * candidate is tried. Returns the successful call with the model that
 * answered it and the refusals it cost, or null when every candidate refused.
 */
async function serveWithFailover(
  candidates: ModelRow[],
  gatewayBody: (model: string) => Json,
): Promise<{ upstream: Response; model: string; failures: Failure[] } | null> {
  const failures: Failure[] = [];
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, gatewayBody(candidate.id));
    } catch (err) {
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      cooled.add(candidate.id);
      log(`gateway unreachable on ${candidate.id}: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) {
      return { upstream, model: candidate.id, failures };
    }
    const text = await upstream.text().catch(() => '');
    failures.push({ status: upstream.status, code: errorCodeOf(text), model: candidate.id });
    cooled.add(candidate.id);
    log(`gateway refused ${candidate.id} with HTTP ${upstream.status}`);
  }
  return null;
}

/** A prompt every candidate refused, as a 502 in the client's wire. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: Failure[]): void {
  const last = failures[failures.length - 1];
  const detail = last !== undefined ? ` (last: HTTP ${last.status} on ${last.model ?? 'unknown'})` : '';
  if (wire === 'anthropic') {
    sendError(res, wire, 502, 'api_error', `every free model refused this prompt${detail}`);
  } else {
    sendError(res, wire, 502, 'upstream_error', `every free model refused this prompt${detail}`);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  }
  const served = await serveWithFailover(candidates, (model) => ({ ...body, model }));
  if (served === null) return refused(res, 'openai', []);
  const { upstream, model, failures } = served;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failuresHeader(failures),
    });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body as ReadableStream<Uint8Array>, model)) {
        res.write(frame);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJsonWithFailures(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), failures);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = candidatesFor(requirementsFor(body));
  if (candidates.length === 0) {
    return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  }
  const served = await serveWithFailover(candidates, (model) => anthropicToOpenAI(body, model));
  if (served === null) return refused(res, 'anthropic', []);
  const { upstream, model, failures } = served;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failuresHeader(failures),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body as ReadableStream<Uint8Array>, model)) {
        res.write(event);
      }
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await upstream.json();
  sendJsonWithFailures(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

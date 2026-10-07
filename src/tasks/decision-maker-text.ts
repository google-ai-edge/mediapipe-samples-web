/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Text playground for the Decision Maker: ask a Boolean, Choice or Score
 * question about any input text. Samples match the Android sample app and are
 * fully editable; edits are kept per question type.
 */

import textTemplate from '../templates/decision-maker-text.html?raw';
import { ViewToggle } from '../components/view-toggle';
import { parseDecisionRequest, toRequestJson, type Draft, type Item, type QuestionKind } from './decision-maker-json';

export type { QuestionKind };

/**
 * Runs one evaluation in the worker; resolves with the raw task result.
 * `schema` evaluates a whole ClassifierSchema with DecisionMaker.evaluate().
 */
export type Evaluator = (kind: QuestionKind | 'schema', text: string, question: object) => Promise<any>;

/** Fields not used by a question type. */
const EMPTY: Omit<Draft, 'input'> = {
  condition: '',
  trueDescription: '',
  falseDescription: '',
  context: '',
  threshold: 0.5,
  items: [],
  instructions: '',
};

/** Laya's built-in descriptions, used when one side is left empty. */
const DEFAULT_TRUE = 'yes, the statement holds';
const DEFAULT_FALSE = 'no, the statement does not hold';

/** Shared domain context for the email-routing samples. */
const EMAIL_CONTEXT =
  'You are an enterprise email security gateway and smart inbox router inspecting an incoming email ' +
  'to detect unsolicited spam or scams and route it to the right folder.';

/** Sample emails used as inputs for the email-routing samples. */
const EMAIL_INPUTS: string[] = [
  'CONGRATULATIONS! You have been selected to claim a $5,000 Cash Prize! Click here immediately to wire your processing fee.',
  'Hi team, attached are the Q3 engineering OKR slides for tomorrow morning review.',
  'FLASH SALE: 80% off luxury watches today only! Unsubscribe at bottom.',
  'Your AWS invoice for August ($142.18) is now available in the billing console.',
];

/** Editable samples; the first one of each type is shown by default. */
const SAMPLES: Record<QuestionKind, { name: string; draft: Draft }[]> = {
  boolean: [
    {
      name: 'Spam detection',
      draft: {
        ...EMPTY,
        input: EMAIL_INPUTS[0],
        condition:
          'Is this email unsolicited commercial bulk marketing, junk mail, a deceptive prize scam, or a fraudulent solicitation?',
        context: EMAIL_CONTEXT,
      },
    },
  ],
  choice: [
    {
      name: 'Inbox folder',
      draft: {
        ...EMPTY,
        input: EMAIL_INPUTS[3],
        items: [
          {
            label: 'primary',
            description: 'Direct personal or work communication between colleagues, teammates, or clients',
          },
          {
            label: 'promotions',
            description: 'Commercial marketing newsletters, flash sales, discount offers, or promotional campaigns',
          },
          {
            label: 'transactional',
            description:
              'Automated account receipts, cloud billing invoices, shipping confirmations, or account alerts',
          },
          {
            label: 'spam_quarantine',
            description: 'Deceptive cash prize scams, advance-fee wire fraud, lottery lures, or abusive junk mail',
          },
        ],
        instructions:
          'Which inbox destination folder should this email be delivered to based on its sender intent and content?',
        context: EMAIL_CONTEXT,
      },
    },
    {
      name: 'Support ticket',
      draft: {
        ...EMPTY,
        input: 'My package never arrived even though tracking says delivered.',
        items: [
          { label: 'shipping', description: 'Delivery problems, lost packages, tracking' },
          { label: 'billing', description: 'Payment issues, duplicate charges, invoices' },
          { label: 'technical', description: 'App crashes, login errors, bugs' },
        ],
        instructions: 'Which department should handle this ticket?',
      },
    },
    {
      name: 'Refund decision',
      draft: {
        ...EMPTY,
        input: "The item is fine, I just don't like it. Refund please.",
        items: [
          { label: 'approve', description: 'item arrived damaged, defective or wrong' },
          { label: 'deny', description: 'no reason given, changed mind, or customer damaged it' },
        ],
        instructions: 'Should this refund request be approved or denied?',
      },
    },
  ],
  // Short, concrete level descriptions give much better scores than bare labels.
  score: [
    {
      name: 'Annoyance score',
      draft: {
        ...EMPTY,
        input: EMAIL_INPUTS[2],
        items: [
          { label: '1', description: 'Clean and expected personal, team, or automated transactional communication' },
          { label: '2', description: 'Mild opt-in commercial update or routine promotional newsletter' },
          { label: '3', description: 'Aggressive unsolicited retail marketing, flash sale hype, or bulk advertising' },
          { label: '4', description: 'Blatant fraudulent scam, fake cash prize lure, or malicious junk spam' },
        ],
        instructions:
          'Rate how spammy, intrusive, or deceptive this email is from 1 (clean work/personal email) to 4 (blatant scam or junk spam).',
        context: EMAIL_CONTEXT,
      },
    },
    {
      name: 'Customer satisfaction',
      draft: {
        ...EMPTY,
        input: 'The support agent was friendly and fixed my issue fast.',
        items: [
          { label: 'Very dissatisfied', description: 'angry, problem not solved, terrible service' },
          { label: 'Dissatisfied', description: 'slow or unhelpful support, problem partly solved' },
          { label: 'Neutral', description: 'okay, average, nothing special' },
          { label: 'Satisfied', description: 'helpful support, problem solved' },
          { label: 'Very satisfied', description: 'excellent, fast, friendly support, delighted' },
        ],
        instructions: 'Rate how satisfied the customer is with the support experience.',
      },
    },
  ],
};

interface Bar {
  label: string;
  probability: number;
  highlighted: boolean;
}

interface Outcome {
  /** Shown above the headline when several questions are evaluated together. */
  title?: string;
  headline: string;
  subtitle?: string;
  bars: Bar[];
}

/** One-line explanation of each question type, shown under the tabs. */
const KIND_HELP: Record<QuestionKind, string> = {
  boolean: 'Boolean: is the condition true for the input text? The model answers Yes or No, with a probability.',
  choice: 'Choice: which option fits the input text best? The model picks one option and scores all of them.',
  score: 'Score: where does the input text fall on a scale? The model picks a level from your rubric.',
};
const JSON_HELP =
  'JSON: paste a request from scratch. A ClassifierSchema with several questions is evaluated in one call.';
const COMBINED_HELP =
  'Combined: define several Boolean / Choice / Score questions and evaluate them together on one input, in one call.';

// ---------------------------------------------------------------------------
// Combined: a ClassifierSchema edited as a form
// ---------------------------------------------------------------------------

type SchemaType = 'binary' | 'categorical' | 'ordinal';

const SCHEMA_TYPES: { value: SchemaType; label: string }[] = [
  { value: 'binary', label: 'Binary (Yes/No)' },
  { value: 'categorical', label: 'Categorical (Choice)' },
  { value: 'ordinal', label: 'Ordinal (1..N Scale)' },
];

/** Accepts the playground names too (boolean / choice / score). */
const SCHEMA_TYPE_ALIASES: Record<string, SchemaType> = {
  binary: 'binary',
  boolean: 'binary',
  categorical: 'categorical',
  choice: 'categorical',
  ordinal: 'ordinal',
  score: 'ordinal',
};

interface SchemaQuestion {
  id: string;
  type: SchemaType;
  prompt: string;
  /** Binary only. */
  threshold: number;
  /** Categorical / ordinal (ordinal: lowest to highest). */
  options: Item[];
}

interface SchemaDraft {
  input: string;
  context: string;
  questions: SchemaQuestion[];
}

/** The email-routing samples as one schema (default for Combined and the JSON example). */
function combinedSample(): SchemaDraft {
  const [binary, categorical, ordinal] = [SAMPLES.boolean[0].draft, SAMPLES.choice[0].draft, SAMPLES.score[0].draft];
  return structuredClone({
    input: EMAIL_INPUTS[0],
    context: EMAIL_CONTEXT,
    questions: [
      { id: 'is_spam', type: 'binary', prompt: binary.condition, threshold: 0.5, options: [] },
      {
        id: 'inbox_folder',
        type: 'categorical',
        prompt: categorical.instructions,
        threshold: 0.5,
        options: categorical.items,
      },
      { id: 'annoyance_score', type: 'ordinal', prompt: ordinal.instructions, threshold: 0.5, options: ordinal.items },
    ],
  });
}

/** SchemaDraft -> ClassifierSchema (without the input). Throws with a readable message. */
function toSchema(d: SchemaDraft): { context?: string; questions: object[] } {
  if (!d.questions.length) throw new Error('Add at least one question.');
  const ids = new Set<string>();
  const questions = d.questions.map((q, i) => {
    const id = q.id.trim();
    const n = `Question ${i + 1}`;
    if (!id) throw new Error(`${n} needs an id.`);
    if (ids.has(id)) throw new Error(`Question id "${id}" is used twice.`);
    ids.add(id);
    if (!q.prompt.trim()) throw new Error(`${n} (${id}) needs a prompt.`);
    if (q.type === 'binary') return { id, type: q.type, prompt: q.prompt.trim(), threshold: q.threshold };
    const options = q.options
      .filter((o) => o.label.trim())
      .map((o) => ({ label: o.label.trim(), description: o.description.trim() }));
    if (options.length < 2) throw new Error(`${n} (${id}) needs at least two options.`);
    return { id, type: q.type, prompt: q.prompt.trim(), options };
  });
  return { ...(d.context.trim() ? { context: d.context.trim() } : {}), questions };
}

/** ClassifierSchema JSON object -> SchemaDraft. */
function fromSchema(input: string, schema: any): SchemaDraft {
  if (!Array.isArray(schema.questions) || !schema.questions.length)
    throw new Error('"questions" must be a non-empty list.');
  return {
    input,
    context: String(schema.context ?? ''),
    questions: schema.questions.map((q: any, i: number) => {
      const type = SCHEMA_TYPE_ALIASES[String(q?.type ?? '').toLowerCase()];
      if (!type) throw new Error(`questions[${i}] has unknown type "${q?.type}". Use binary, categorical, or ordinal.`);
      const options: Item[] = (Array.isArray(q.options) ? q.options : []).map((o: any) =>
        typeof o === 'string'
          ? { label: o, description: '' }
          : { label: String(o?.label ?? ''), description: String(o?.description ?? '') }
      );
      return {
        id: String(q.id ?? ''),
        type,
        prompt: String(q.prompt ?? ''),
        threshold: typeof q.threshold === 'number' ? q.threshold : 0.5,
        options: type === 'binary' ? [] : options,
      };
    }),
  };
}

/** "Insert example" for the JSON tab. */
function jsonExample(): string {
  const d = combinedSample();
  return JSON.stringify({ input: d.input, ...toSchema(d) }, null, 2);
}

export class DecisionTextPlayground {
  /** Question type of the form editor (kept while another tab is open). */
  private kind: QuestionKind = 'boolean';
  /** Which editor is shown: the single-question form, Combined, or JSON. */
  private view: 'form' | 'combined' | 'json' = 'form';
  private combined: SchemaDraft = combinedSample();
  private sampleIndex: Record<QuestionKind, number> = { boolean: 0, choice: 0, score: 0 };
  private drafts: Record<QuestionKind, Draft> = {
    boolean: structuredClone(SAMPLES.boolean[0].draft),
    choice: structuredClone(SAMPLES.choice[0].draft),
    score: structuredClone(SAMPLES.score[0].draft),
  };
  private ready = false;
  private busy = false;
  private el: Record<string, HTMLElement> = {};
  private kindToggle!: ViewToggle;

  constructor(
    private root: HTMLElement,
    private evaluate: Evaluator,
    private onStatus: (text: string, inferenceTime?: number) => void
  ) {}

  init() {
    this.root.innerHTML = textTemplate;
    this.root.querySelectorAll<HTMLElement>('[id]').forEach((node) => (this.el[node.id] = node));
    (this.el['dt-true-desc'] as HTMLTextAreaElement).placeholder = DEFAULT_TRUE;
    (this.el['dt-false-desc'] as HTMLTextAreaElement).placeholder = DEFAULT_FALSE;

    this.kindToggle = new ViewToggle(
      'dt-kind-toggle',
      [
        { label: 'Boolean', value: 'boolean', icon: 'rule' },
        { label: 'Choice', value: 'choice', icon: 'list' },
        { label: 'Score', value: 'score', icon: 'star_half' },
        { label: 'Combined', value: 'combined', icon: 'checklist' },
        { label: 'JSON', value: 'json', icon: 'data_object' },
      ],
      this.kind,
      (value) => {
        if (this.view === 'form') this.saveDraft();
        this.view = value === 'json' || value === 'combined' ? value : 'form';
        if (this.view === 'form') this.kind = value as QuestionKind;
        this.showDraft();
        this.clearResult();
      },
      'tabs'
    );

    this.el['dt-evaluate'].addEventListener('click', () => this.run());
    this.el['dt-add-item'].addEventListener('click', () => {
      this.saveDraft();
      this.drafts[this.kind].items.push({ label: '', description: '' });
      this.renderItems();
    });
    this.el['dt-reset'].addEventListener('click', () => this.loadSample(this.sampleIndex[this.kind]));
    this.el['dt-threshold'].addEventListener('input', (e) => {
      this.el['dt-threshold-value'].textContent = parseFloat((e.target as HTMLInputElement).value).toFixed(2);
    });

    this.initCombinedTab();
    this.initJsonTab();
    this.showDraft();
    this.clearResult();
  }

  setReady(ready: boolean) {
    this.ready = ready;
    this.updateButton();
  }

  // ---------------------------------------------------------------------------
  // Editor
  // ---------------------------------------------------------------------------

  private showDraft() {
    this.el['dt-form-card'].style.display = this.view === 'form' ? '' : 'none';
    this.el['dt-combined-card'].style.display = this.view === 'combined' ? '' : 'none';
    this.el['dt-json-card'].style.display = this.view === 'json' ? '' : 'none';
    if (this.view === 'json') {
      this.el['dt-kind-help'].textContent = JSON_HELP;
      return;
    }
    if (this.view === 'combined') {
      this.el['dt-kind-help'].textContent = COMBINED_HELP;
      this.showCombined();
      return;
    }
    const d = this.drafts[this.kind];
    const isBoolean = this.kind === 'boolean';
    (this.el['dt-input'] as HTMLTextAreaElement).value = d.input;
    (this.el['dt-condition'] as HTMLTextAreaElement).value = d.condition;
    (this.el['dt-true-desc'] as HTMLTextAreaElement).value = d.trueDescription;
    (this.el['dt-false-desc'] as HTMLTextAreaElement).value = d.falseDescription;
    (this.el['dt-context'] as HTMLInputElement).value = d.context;
    (this.el['dt-threshold'] as HTMLInputElement).value = `${d.threshold}`;
    this.el['dt-threshold-value'].textContent = d.threshold.toFixed(2);
    (this.el['dt-instructions'] as HTMLInputElement).value = d.instructions;
    this.el['dt-boolean-fields'].style.display = isBoolean ? '' : 'none';
    this.el['dt-list-fields'].style.display = isBoolean ? 'none' : '';
    this.el['dt-items-title'].textContent = this.kind === 'score' ? 'Rubric' : 'Options';
    this.el['dt-items-hint'].textContent =
      this.kind === 'score'
        ? 'levels from lowest to highest, each with a short description'
        : 'the answers to pick from, each with a short description';
    this.el['dt-add-label'].textContent = this.kind === 'score' ? 'Add level' : 'Add option';
    this.el['dt-kind-help'].textContent = KIND_HELP[this.kind];
    this.renderItems();
    this.renderSamples();
  }

  /** Sample chips (only shown when a type has more than one sample). */
  private renderSamples() {
    const box = this.el['dt-samples'];
    box.innerHTML = '';
    const samples = SAMPLES[this.kind];
    if (samples.length < 2) return;
    samples.forEach((sample, i) => {
      const chip = document.createElement('button');
      chip.className = `dt-sample ${i === this.sampleIndex[this.kind] ? 'active' : ''}`;
      chip.textContent = sample.name;
      chip.addEventListener('click', () => this.loadSample(i));
      box.appendChild(chip);
    });
  }

  private loadSample(i: number) {
    this.sampleIndex[this.kind] = i;
    this.drafts[this.kind] = structuredClone(SAMPLES[this.kind][i].draft);
    this.showDraft();
    this.clearResult();
  }

  private renderItems() {
    const list = this.el['dt-items'];
    list.innerHTML = '';
    const isScore = this.kind === 'score';
    this.drafts[this.kind].items.forEach((item, i) => {
      const row = document.createElement('div');
      row.className = 'dt-item';
      row.innerHTML = `
        <input class="dt-field dt-item-label" placeholder="${isScore ? `Level ${i + 1}` : 'Key'}" />
        <input class="dt-field dt-item-desc" placeholder="Description" />
        <button class="dt-icon-btn" title="Remove"><span class="material-icons">close</span></button>`;
      (row.querySelector('.dt-item-label') as HTMLInputElement).value = item.label;
      (row.querySelector('.dt-item-desc') as HTMLInputElement).value = item.description;
      row.querySelector('button')!.addEventListener('click', () => {
        this.saveDraft();
        this.drafts[this.kind].items.splice(i, 1);
        this.renderItems();
      });
      list.appendChild(row);
    });
  }

  private saveDraft() {
    const d = this.drafts[this.kind];
    d.input = (this.el['dt-input'] as HTMLTextAreaElement).value;
    d.condition = (this.el['dt-condition'] as HTMLTextAreaElement).value;
    d.trueDescription = (this.el['dt-true-desc'] as HTMLTextAreaElement).value;
    d.falseDescription = (this.el['dt-false-desc'] as HTMLTextAreaElement).value;
    d.context = (this.el['dt-context'] as HTMLInputElement).value;
    d.threshold = parseFloat((this.el['dt-threshold'] as HTMLInputElement).value);
    d.instructions = (this.el['dt-instructions'] as HTMLInputElement).value;
    d.items = [...this.el['dt-items'].querySelectorAll('.dt-item')].map((row) => ({
      label: (row.querySelector('.dt-item-label') as HTMLInputElement).value,
      description: (row.querySelector('.dt-item-desc') as HTMLInputElement).value,
    }));
  }

  private updateButton() {
    (this.el['dt-evaluate'] as HTMLButtonElement).disabled = !this.ready || this.busy;
    (this.el['dt-json-evaluate'] as HTMLButtonElement).disabled = !this.ready || this.busy;
    (this.el['dt-c-evaluate'] as HTMLButtonElement).disabled = !this.ready || this.busy;
  }

  // ---------------------------------------------------------------------------
  // Combined tab: several questions on one input, edited as a form
  // ---------------------------------------------------------------------------

  private initCombinedTab() {
    this.el['dt-c-input'].addEventListener('input', (e) => {
      this.combined.input = (e.target as HTMLTextAreaElement).value;
    });
    this.el['dt-c-context'].addEventListener('input', (e) => {
      this.combined.context = (e.target as HTMLInputElement).value;
    });
    this.el['dt-c-add-question'].addEventListener('click', () => {
      this.combined.questions.push({
        id: `q${this.combined.questions.length + 1}`,
        type: 'binary',
        prompt: '',
        threshold: 0.5,
        options: [],
      });
      this.renderQuestions();
    });
    this.el['dt-c-reset'].addEventListener('click', () => {
      this.combined = combinedSample();
      this.showCombined();
      this.clearResult();
    });
    this.el['dt-c-open-json'].addEventListener('click', () => {
      const d = this.combined;
      const schema = { context: d.context, ...this.looseSchema() };
      (this.el['dt-json-text'] as HTMLTextAreaElement).value = JSON.stringify({ input: d.input, ...schema }, null, 2);
      this.el['dt-json-error'].textContent = '';
      this.kindToggle.setActive('json');
    });
    this.el['dt-c-evaluate'].addEventListener('click', () => this.runCombined());
  }

  /** Current Combined questions as JSON, without validation (for Edit as JSON). */
  private looseSchema() {
    return {
      questions: this.combined.questions.map((q) =>
        q.type === 'binary'
          ? { id: q.id, type: q.type, prompt: q.prompt, threshold: q.threshold }
          : { id: q.id, type: q.type, prompt: q.prompt, options: q.options }
      ),
    };
  }

  private showCombined() {
    (this.el['dt-c-input'] as HTMLTextAreaElement).value = this.combined.input;
    (this.el['dt-c-context'] as HTMLInputElement).value = this.combined.context;
    this.renderQuestions();
  }

  /** Re-renders the question list (inputs write straight into this.combined). */
  private renderQuestions() {
    const box = this.el['dt-c-questions'];
    box.innerHTML = '';
    this.combined.questions.forEach((q, qi) => {
      const block = document.createElement('div');
      block.className = 'dt-q';
      block.innerHTML = `
        <div class="dt-q-head">
          <span class="dt-q-badge"></span>
          <input class="dt-field dt-q-id" placeholder="id (result key)" />
          <select class="dt-field dt-q-type">
            ${SCHEMA_TYPES.map((t) => `<option value="${t.value}">${t.label}</option>`).join('')}
          </select>
          <button class="dt-icon-btn dt-q-remove" title="Remove question"><span class="material-icons">close</span></button>
        </div>
        <input class="dt-field dt-q-prompt" />
        <div class="dt-q-body"></div>`;
      block.querySelector('.dt-q-badge')!.textContent = `Q${qi + 1}: ${q.type.toUpperCase()}`;

      const id = block.querySelector('.dt-q-id') as HTMLInputElement;
      id.value = q.id;
      id.addEventListener('input', () => (q.id = id.value));

      const type = block.querySelector('.dt-q-type') as HTMLSelectElement;
      type.value = q.type;
      type.addEventListener('change', () => {
        q.type = type.value as SchemaType;
        while (q.type !== 'binary' && q.options.length < 2) q.options.push({ label: '', description: '' });
        this.renderQuestions();
      });

      block.querySelector('.dt-q-remove')!.addEventListener('click', () => {
        this.combined.questions.splice(qi, 1);
        this.renderQuestions();
      });

      const prompt = block.querySelector('.dt-q-prompt') as HTMLInputElement;
      prompt.value = q.prompt;
      prompt.placeholder =
        q.type === 'binary'
          ? 'Condition, e.g. Is this email spam?'
          : q.type === 'ordinal'
            ? 'Instructions, e.g. Rate how urgent this email is.'
            : 'Instructions, e.g. Which folder should this email go to?';
      prompt.addEventListener('input', () => (q.prompt = prompt.value));

      const body = block.querySelector('.dt-q-body') as HTMLElement;
      if (q.type === 'binary') {
        body.innerHTML = `
          <div class="dt-q-threshold">
            <span>Threshold</span>
            <input type="range" min="0.05" max="0.95" step="0.05" class="range-slider" />
            <b></b>
          </div>`;
        const range = body.querySelector('input') as HTMLInputElement;
        const value = body.querySelector('b')!;
        range.value = `${q.threshold}`;
        value.textContent = q.threshold.toFixed(2);
        range.addEventListener('input', () => {
          q.threshold = parseFloat(range.value);
          value.textContent = q.threshold.toFixed(2);
        });
      } else {
        q.options.forEach((o, oi) => {
          const row = document.createElement('div');
          row.className = 'dt-item';
          row.innerHTML = `
            <input class="dt-field dt-item-label" />
            <input class="dt-field dt-item-desc" placeholder="Description" />
            <button class="dt-icon-btn" title="Remove"><span class="material-icons">close</span></button>`;
          const label = row.querySelector('.dt-item-label') as HTMLInputElement;
          const desc = row.querySelector('.dt-item-desc') as HTMLInputElement;
          label.placeholder = q.type === 'ordinal' ? `Level ${oi + 1}` : 'Key';
          label.value = o.label;
          desc.value = o.description;
          label.addEventListener('input', () => (o.label = label.value));
          desc.addEventListener('input', () => (o.description = desc.value));
          row.querySelector('button')!.addEventListener('click', () => {
            q.options.splice(oi, 1);
            this.renderQuestions();
          });
          body.appendChild(row);
        });
        const add = document.createElement('button');
        add.className = 'dt-link-btn';
        add.innerHTML = `<span class="material-icons">add</span>${q.type === 'ordinal' ? 'Add level' : 'Add option'}`;
        add.addEventListener('click', () => {
          q.options.push({ label: '', description: '' });
          this.renderQuestions();
        });
        body.appendChild(add);
      }
      box.appendChild(block);
    });
  }

  /** Evaluates all Combined questions on the input with one evaluate(input, schema) call. */
  private runCombined() {
    const input = this.combined.input.trim();
    if (!input) return this.onStatus('Enter some input text.');
    let schema: { context?: string; questions: object[] };
    try {
      schema = toSchema(this.combined);
    } catch (e: any) {
      return this.onStatus(e?.message ?? String(e));
    }
    this.runSchema(input, schema);
  }

  /** Runs a ClassifierSchema in the worker and shows one result block per question. */
  private async runSchema(input: string, schema: { questions: any[] }) {
    this.busy = true;
    this.updateButton();
    this.onStatus('Evaluating...');
    try {
      const msg = await this.evaluate('schema', input, schema);
      if (msg.type !== 'DECIDE_RESULT') throw new Error(msg.error);
      this.showOutcomes(schema.questions.map((q) => this.formatDecision(q, msg.result?.[q.id])));
      this.onStatus('Done', msg.inferenceTime);
    } catch (e: any) {
      this.onStatus(`Error: ${e?.message ?? e}`);
    } finally {
      this.busy = false;
      this.updateButton();
    }
  }

  // ---------------------------------------------------------------------------
  // JSON tab: paste a Decision API request instead of filling in the form
  // ---------------------------------------------------------------------------

  private initJsonTab() {
    const textarea = this.el['dt-json-text'] as HTMLTextAreaElement;
    const error = this.el['dt-json-error'];
    const setText = (text: string) => {
      textarea.value = text;
      error.textContent = '';
      this.clearResult();
    };

    // From the form: open the current question as JSON.
    this.el['dt-open-json'].addEventListener('click', () => {
      this.saveDraft();
      const d = this.drafts[this.kind];
      setText(toRequestJson(this.kind, d.input.trim(), this.buildQuestion(this.kind, d).question));
      this.kindToggle.setActive('json');
      textarea.scrollTop = 0;
    });
    this.el['dt-json-example'].addEventListener('click', () => setText(jsonExample()));
    this.el['dt-json-clear'].addEventListener('click', () => {
      setText('');
      textarea.focus();
    });
    this.el['dt-json-to-form'].addEventListener('click', () => {
      try {
        const obj = JSON.parse(textarea.value);
        const schema = obj?.questions ? obj : obj?.schema?.questions ? obj.schema : undefined;
        if (schema) {
          this.combined = fromSchema(String(obj.input ?? obj.text ?? ''), schema);
          this.kindToggle.setActive('combined');
          return;
        }
        const { kind, draft, note } = parseDecisionRequest(textarea.value, EMPTY);
        this.drafts[kind] = draft;
        this.kindToggle.setActive(kind); // switches tab, shows the new draft
        if (note) this.onStatus(note);
      } catch (e: any) {
        error.textContent = e instanceof SyntaxError ? `Invalid JSON: ${e.message}` : (e?.message ?? String(e));
      }
    });
    this.el['dt-json-evaluate'].addEventListener('click', () => this.runJson());
  }

  /** Evaluates the JSON tab: a whole ClassifierSchema in one call, or a single question. */
  private async runJson(): Promise<void> {
    const error = this.el['dt-json-error'];
    error.textContent = '';
    const json = (this.el['dt-json-text'] as HTMLTextAreaElement).value;
    if (!json.trim()) {
      error.textContent = 'Paste a request, or click "Insert example".';
      return;
    }

    let task: () => Promise<{ outcomes: Outcome[]; inferenceTime: number }>;
    try {
      const obj = JSON.parse(json);
      const schema = obj?.questions ? obj : obj?.schema?.questions ? obj.schema : undefined;
      if (schema) {
        const input = String(obj.input ?? obj.text ?? '').trim();
        if (!input) throw new Error('Add the text to evaluate as "input".');
        // Same checks as the Combined tab (ids, prompts, options); the request is sent as pasted.
        toSchema(fromSchema(input, schema));
        const { input: _i, text: _t, schema: _s, ...schemaOnly } = schema === obj ? obj : { ...schema };
        task = async () => {
          const msg = await this.evaluate('schema', input, schemaOnly);
          if (msg.type !== 'DECIDE_RESULT') throw new Error(msg.error);
          const outcomes = schema.questions.map((q: any) => this.formatDecision(q, msg.result?.[q.id]));
          return { outcomes, inferenceTime: msg.inferenceTime };
        };
      } else {
        const { kind, draft } = parseDecisionRequest(json, EMPTY);
        if (!draft.input.trim()) throw new Error('Add the text to evaluate as "text".');
        const { question, error: invalid } = this.buildQuestion(kind, draft);
        if (invalid) throw new Error(invalid);
        task = async () => {
          const msg = await this.evaluate(kind, draft.input.trim(), question);
          if (msg.type !== 'DECIDE_RESULT') throw new Error(msg.error);
          return { outcomes: [this.format(kind, msg.result, draft)], inferenceTime: msg.inferenceTime };
        };
      }
    } catch (e: any) {
      error.textContent = e instanceof SyntaxError ? `Invalid JSON: ${e.message}` : (e?.message ?? String(e));
      return;
    }

    this.busy = true;
    this.updateButton();
    this.onStatus('Evaluating...');
    try {
      const { outcomes, inferenceTime } = await task();
      this.showOutcomes(outcomes);
      this.onStatus('Done', inferenceTime);
    } catch (e: any) {
      this.onStatus(`Error: ${e?.message ?? e}`);
    } finally {
      this.busy = false;
      this.updateButton();
    }
  }

  /** One ClassifierDecision from evaluate(input, schema), shown like the form results. */
  private formatDecision(q: any, d: any): Outcome {
    const title = `${q.id} · ${q.type ?? ''}`;
    if (!d) return { title, headline: '-', subtitle: 'No result for this question.', bars: [] };
    const probs: { label: string; probability: number }[] = d.probabilities ?? [];
    if (d.probability !== undefined) {
      const yes = d.label === 'true';
      return {
        title,
        headline: yes ? 'Yes' : 'No',
        subtitle: `P(yes) = ${d.probability.toFixed(2)} · threshold ${(q.threshold ?? 0.5).toFixed(2)}`,
        bars: [
          { label: 'Yes', probability: d.probability, highlighted: yes },
          { label: 'No', probability: 1 - d.probability, highlighted: !yes },
        ],
      };
    }
    if (d.expectedScore !== undefined) {
      // The library's expectedScore counts levels from 0; show 1..N like the Score tab.
      const expected = probs.reduce((sum, p, i) => sum + (i + 1) * p.probability, 0);
      return {
        title,
        headline: `${expected.toFixed(2)} / ${probs.length}`,
        subtitle: `Most likely: ${d.label}`,
        bars: probs.map((p) => ({ label: p.label, probability: p.probability, highlighted: p.label === d.label })),
      };
    }
    const desc = (q.options ?? []).find((o: any) => o?.label === d.label)?.description;
    return {
      title,
      headline: d.label,
      subtitle: desc || `confidence ${(d.confidence ?? 0).toFixed(2)}`,
      bars: [...probs]
        .sort((a, b) => b.probability - a.probability)
        .map((p) => ({ label: p.label, probability: p.probability, highlighted: p.label === d.label })),
    };
  }

  // ---------------------------------------------------------------------------
  // Evaluation
  // ---------------------------------------------------------------------------

  /** Builds the question in the same shape as the Android sample. */
  private buildQuestion(kind: QuestionKind, d: Draft): { question: object; error?: string } {
    const context = d.context.trim() ? { context: d.context.trim() } : {};
    if (kind === 'boolean') {
      if (!d.condition.trim()) return { question: {}, error: 'Enter a condition.' };
      // Custom meanings for false / true; an empty side falls back to Laya's default.
      const t = d.trueDescription.trim();
      const f = d.falseDescription.trim();
      const options =
        t || f
          ? {
              options: [
                { label: 'false', description: f || DEFAULT_FALSE },
                { label: 'true', description: t || DEFAULT_TRUE },
              ],
            }
          : {};
      return { question: { condition: d.condition.trim(), threshold: d.threshold, ...options, ...context } };
    }
    const items = d.items.filter((it) => it.label.trim());
    if (items.length < 2) return { question: {}, error: 'Add at least two entries.' };
    if (kind === 'choice') {
      const criteria = Object.fromEntries(items.map((it) => [it.label.trim(), it.description.trim()]));
      return { question: { criteria, instructions: d.instructions.trim(), ...context } };
    }
    // Each rubric level is sent as "name: description".
    const rubric = items.map((it) =>
      it.description.trim() ? `${it.label.trim()}: ${it.description.trim()}` : it.label.trim()
    );
    return { question: { rubric, instructions: d.instructions.trim(), ...context } };
  }

  /** Evaluates the current form; resolves true if a result was shown. */
  private async run(): Promise<boolean> {
    this.saveDraft();
    const d = this.drafts[this.kind];
    if (!d.input.trim()) {
      this.onStatus('Enter some input text.');
      return false;
    }
    const { question, error } = this.buildQuestion(this.kind, d);
    if (error) {
      this.onStatus(error);
      return false;
    }

    this.busy = true;
    this.updateButton();
    this.onStatus('Evaluating...');
    try {
      const msg = await this.evaluate(this.kind, d.input.trim(), question);
      if (msg.type !== 'DECIDE_RESULT') throw new Error(msg.error);
      this.showOutcomes([this.format(this.kind, msg.result, d)]);
      this.onStatus('Done', msg.inferenceTime);
      return true;
    } catch (e: any) {
      this.onStatus(`Error: ${e?.message ?? e}`);
      return false;
    } finally {
      this.busy = false;
      this.updateButton();
    }
  }

  private format(kind: QuestionKind, r: any, d: Draft): Outcome {
    if (kind === 'boolean') {
      return {
        headline: r.value ? 'Yes' : 'No',
        subtitle: `P(yes) = ${r.probabilityTrue.toFixed(2)} · threshold ${d.threshold.toFixed(2)}`,
        bars: [
          { label: 'Yes', probability: r.probabilityTrue, highlighted: r.value },
          { label: 'No', probability: 1 - r.probabilityTrue, highlighted: !r.value },
        ],
      };
    }
    if (kind === 'choice') {
      const desc = d.items.find((it) => it.label.trim() === r.selectedKey)?.description;
      return {
        headline: r.selectedKey,
        subtitle: desc || undefined,
        bars: Object.entries(r.probabilities as Record<string, number>)
          .sort((a, b) => b[1] - a[1])
          .map(([label, p]) => ({ label, probability: p, highlighted: label === r.selectedKey })),
      };
    }
    // Score: expected score on a 1..N scale, like the Android sample.
    const probs: number[] = r.probabilities;
    const levels = d.items.filter((it) => it.label.trim()).map((it) => it.label.trim());
    const expected = probs.reduce((sum, p, i) => sum + (i + 1) * p, 0);
    const best = probs.indexOf(Math.max(...probs));
    return {
      headline: `${expected.toFixed(2)} / ${probs.length}`,
      subtitle: levels[best] ? `Most likely: ${levels[best]}` : undefined,
      bars: probs.map((p, i) => ({
        // Skip the name when the level is just its number (e.g. "1".."4").
        label: levels[i] && levels[i] !== `${i + 1}` ? `${i + 1} · ${levels[i]}` : `${i + 1}`,
        probability: p,
        highlighted: i === best,
      })),
    };
  }

  private showOutcomes(outcomes: Outcome[]) {
    const box = this.el['dt-results'];
    box.innerHTML = '';
    for (const o of outcomes) {
      const block = document.createElement('div');
      block.className = 'dt-result';
      block.innerHTML = `
        <div class="dt-result-title"></div>
        <div class="dt-headline"></div>
        <div class="dt-subtitle"></div>
        <div class="dt-bars"></div>`;
      const title = block.querySelector<HTMLElement>('.dt-result-title')!;
      title.textContent = o.title ?? '';
      title.style.display = o.title ? '' : 'none';
      block.querySelector('.dt-headline')!.textContent = o.headline;
      block.querySelector('.dt-subtitle')!.textContent = o.subtitle ?? '';
      const bars = block.querySelector('.dt-bars')!;
      for (const bar of o.bars) {
        const p = Math.min(1, Math.max(0, bar.probability));
        const row = document.createElement('div');
        row.className = `dt-bar ${bar.highlighted ? 'hl' : ''}`;
        row.innerHTML = `
          <span class="dt-bar-label"></span>
          <div class="dt-bar-track"><div class="dt-bar-fill" style="width: ${p * 100}%"></div></div>
          <span class="dt-bar-value">${(p * 100).toFixed(1)}%</span>`;
        row.querySelector('.dt-bar-label')!.textContent = bar.label;
        bars.appendChild(row);
      }
      box.appendChild(block);
    }
  }

  private clearResult() {
    this.el['dt-results'].innerHTML = '<div class="dt-headline dt-muted">-</div>';
  }
}

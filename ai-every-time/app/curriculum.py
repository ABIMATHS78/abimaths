"""Learning paths and their lessons.

Each lesson carries objectives and vetted key points. Companions teach the lesson
in their own voice, but the key points anchor every lesson, quiz, flashcard deck
and exam to the same accurate core, whichever companion is teaching.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lesson:
    id: str
    title: str
    objectives: tuple[str, ...]
    key_points: tuple[str, ...]
    minutes: int = 10


@dataclass(frozen=True)
class LearningPath:
    id: str
    title: str
    description: str
    level: str
    topic: str  # one of persona.EXPERTISE_AREAS, used to suggest a companion
    lessons: tuple[Lesson, ...] = field(default_factory=tuple)


PATHS: tuple[LearningPath, ...] = (
    LearningPath(
        id="ai-foundations",
        title="AI Foundations",
        description="What AI is, how machines learn from data, and how we know whether a model is any good.",
        level="beginner",
        topic="Machine Learning Foundations",
        lessons=(
            Lesson(
                "what-is-ai", "What AI actually is",
                ("Explain AI, machine learning and deep learning and how they relate",
                 "Tell narrow AI apart from the idea of general AI"),
                ("Artificial intelligence is the broad goal of building systems that perform tasks normally needing human intelligence.",
                 "Machine learning is a subset of AI where systems learn patterns from data instead of following hand-written rules.",
                 "Deep learning is a subset of machine learning that uses neural networks with many layers.",
                 "Today's AI systems are narrow: each is built for particular kinds of tasks rather than general human-level ability."),
            ),
            Lesson(
                "rules-vs-learning", "Rules versus learning from data",
                ("Contrast rule-based programs with learned models", "Describe what a model and its parameters are"),
                ("A rule-based program follows instructions a person wrote explicitly.",
                 "A machine learning model is a function whose parameters are adjusted during training to fit example data.",
                 "Learned models shine when the rules are too complex to write down, such as recognising faces or speech.",
                 "A trained model makes predictions on new inputs it did not see during training; this is called inference."),
            ),
            Lesson(
                "types-of-learning", "Supervised, unsupervised and reinforcement learning",
                ("Name the three main learning paradigms", "Match real problems to the right paradigm"),
                ("Supervised learning trains on inputs paired with correct outputs, called labels.",
                 "Classification predicts a category; regression predicts a number. Both are supervised tasks.",
                 "Unsupervised learning finds structure in unlabelled data, for example clustering similar customers.",
                 "Reinforcement learning trains an agent through rewards and penalties received from acting in an environment."),
            ),
            Lesson(
                "training-and-data", "Training data and how models learn",
                ("Explain why data quality matters", "Describe the training loop at a high level"),
                ("Training repeatedly compares the model's predictions with the correct answers and nudges parameters to reduce the error.",
                 "The error is measured by a loss function; gradient descent adjusts parameters in the direction that lowers the loss.",
                 "A model can only learn patterns present in its training data, so biased or noisy data produces biased or noisy models.",
                 "Data is usually split into training, validation and test sets so performance is measured on unseen examples."),
            ),
            Lesson(
                "overfitting-evaluation", "Overfitting and evaluating models",
                ("Recognise overfitting and underfitting", "Choose sensible evaluation metrics"),
                ("Overfitting is when a model memorises training data and performs poorly on new data.",
                 "Underfitting is when a model is too simple to capture the real pattern, so it does poorly even on training data.",
                 "Accuracy can mislead on imbalanced data; precision and recall show how a model handles the rarer class.",
                 "Evaluating on a held-out test set that was never used for training gives an honest estimate of real-world performance."),
            ),
        ),
    ),
    LearningPath(
        id="llms",
        title="Understanding Large Language Models",
        description="Tokens, transformers and training: how models like Claude and ChatGPT work, and where they fall short.",
        level="intermediate",
        topic="Large Language Models",
        lessons=(
            Lesson(
                "tokens", "Tokens: how models read text",
                ("Explain what a token is", "Describe why token counts matter"),
                ("Language models process text as tokens, which are chunks such as whole words, word pieces or punctuation.",
                 "A tokenizer converts text into a sequence of token IDs before the model sees it.",
                 "Models have a context window: a maximum number of tokens they can consider at once.",
                 "Usage costs and speed of hosted models are typically measured in tokens."),
            ),
            Lesson(
                "next-token", "Predicting the next token",
                ("Describe the core objective of a language model", "Explain how text is generated one token at a time"),
                ("A language model is trained to predict the next token given the tokens before it.",
                 "Generation repeats that prediction, appending each chosen token and predicting again.",
                 "The model outputs a probability for every token in its vocabulary; sampling settings decide how one is chosen.",
                 "Fluent text is not the same as true text: the objective rewards plausible continuations."),
            ),
            Lesson(
                "transformers-attention", "Transformers and attention",
                ("Explain the role of attention", "Describe why transformers scaled so well"),
                ("The transformer architecture, introduced in 2017, is the basis of modern large language models.",
                 "Attention lets each token weigh how relevant every other token in the context is when building its representation.",
                 "Transformers process all tokens of a sequence in parallel during training, which made training on huge datasets practical.",
                 "Embeddings represent tokens as vectors of numbers so that related meanings sit close together."),
            ),
            Lesson(
                "training-llms", "Pretraining, fine-tuning and feedback",
                ("Name the main stages of building an assistant model", "Explain the purpose of each stage"),
                ("Pretraining teaches a model general language patterns by predicting tokens across a very large text corpus.",
                 "Fine-tuning continues training on narrower, curated data to shape the model for particular tasks or behaviour.",
                 "Reinforcement learning from human feedback (RLHF) uses human preference judgements to steer a model toward helpful, harmless answers.",
                 "Alignment work aims to make models behave according to intended values and instructions."),
            ),
            Lesson(
                "llm-limits", "Hallucinations and limitations",
                ("Explain why models hallucinate", "Apply strategies to reduce errors"),
                ("A hallucination is a confident-sounding output that is false or unsupported.",
                 "Models have a training cutoff, so they may not know about recent events unless given that information.",
                 "Retrieval-augmented generation (RAG) supplies relevant documents in the prompt so answers can be grounded in sources.",
                 "Important outputs should be verified, especially facts, figures, citations and code."),
            ),
        ),
    ),
    LearningPath(
        id="prompting-agents",
        title="Prompting and AI Agents",
        description="Write prompts that work, get structured results, and understand how AI agents use tools.",
        level="beginner",
        topic="Prompt Engineering",
        lessons=(
            Lesson(
                "prompt-basics", "Clear prompts: role, goal, context, format",
                ("Write a clear, specific prompt", "Explain why context improves answers"),
                ("Clear prompts state the task, the goal and what a good answer looks like.",
                 "Giving relevant context, such as the audience, constraints and background, produces more useful answers.",
                 "Specifying the output format, for example a table or bullet list, makes results easier to use.",
                 "A system prompt sets overall behaviour and role; user messages carry the specific requests."),
            ),
            Lesson(
                "examples-iteration", "Examples and iteration",
                ("Use examples to steer outputs", "Improve prompts systematically"),
                ("Including a few examples of the desired input and output, called few-shot prompting, helps models match a pattern.",
                 "Asking the model to work through a problem step by step can improve results on reasoning tasks.",
                 "Prompting is iterative: test on several realistic inputs, inspect failures and refine.",
                 "Breaking a complex task into smaller steps, or a chain of prompts, is often more reliable than one giant prompt."),
            ),
            Lesson(
                "structured-output", "Getting structured output",
                ("Request machine-readable output", "Explain why schemas help applications"),
                ("Applications often need outputs in a fixed structure such as JSON.",
                 "A JSON schema describes the fields, types and required properties an output must have.",
                 "Many model APIs offer structured output modes that constrain responses to match a schema.",
                 "Validating outputs in code catches problems before they reach users."),
            ),
            Lesson(
                "tool-use", "Tools: letting models take actions",
                ("Describe how tool use works", "Identify good tool designs"),
                ("With tool use, a developer describes functions the model may call, with names, descriptions and input schemas.",
                 "The model decides when to call a tool and with what inputs; the application runs the tool and returns the result.",
                 "Clear tool names and descriptions strongly affect whether the model uses tools correctly.",
                 "Tools let models fetch fresh data, do exact calculations and act in other systems."),
            ),
            Lesson(
                "agent-loop", "The agent loop",
                ("Explain what makes a system an agent", "Recognise risks and safeguards for agents"),
                ("An AI agent repeatedly chooses an action, uses a tool, observes the result and decides what to do next until the task is done.",
                 "Agents suit open-ended, multi-step tasks; simple, well-defined tasks are often better as a single call or fixed workflow.",
                 "Because agents act autonomously, they need limits such as permissions, budgets and human approval for risky actions.",
                 "Logging each step makes agent behaviour observable and easier to debug."),
            ),
        ),
    ),
    LearningPath(
        id="responsible-ai",
        title="Responsible AI",
        description="Bias, privacy, safety and regulation: building and using AI that people can trust.",
        level="beginner",
        topic="AI Ethics & Safety",
        lessons=(
            Lesson(
                "bias-fairness", "Bias and fairness",
                ("Explain where bias in AI comes from", "Describe ways to detect and reduce it"),
                ("AI systems can reproduce or amplify biases present in their training data or design choices.",
                 "Bias can enter through unrepresentative data, historical inequalities in labels, or the choice of what to optimise.",
                 "Fairness is assessed by comparing a model's performance and outcomes across different groups.",
                 "Diverse teams, representative data and regular audits help reduce harmful bias."),
            ),
            Lesson(
                "privacy-data", "Privacy and data protection",
                ("Identify privacy risks in AI systems", "Apply basic data protection principles"),
                ("AI systems often rely on personal data, which creates risks if it is collected, stored or used carelessly.",
                 "Data minimisation means collecting only the data that is needed for a clear purpose.",
                 "Laws such as the EU's GDPR give people rights over their personal data, including access and deletion.",
                 "Sensitive information should not be pasted into AI tools unless their data handling has been checked."),
            ),
            Lesson(
                "safety-misuse", "Safety, reliability and misuse",
                ("Distinguish accidents from misuse", "Describe common safety practices"),
                ("AI safety covers both unintended failures of a system and deliberate misuse by people.",
                 "Red-teaming means deliberately probing a system to find harmful or unexpected behaviour before release.",
                 "Deepfakes and AI-generated misinformation are prominent misuse risks; checking sources helps counter them.",
                 "Monitoring a system after deployment matters because real-world use differs from testing."),
            ),
            Lesson(
                "regulation", "How AI is regulated",
                ("Explain a risk-based approach to regulation", "Name examples of AI governance"),
                ("Risk-based regulation applies stricter requirements to AI uses that can cause greater harm.",
                 "The EU AI Act sorts AI systems into risk categories, with some practices prohibited and high-risk uses heavily regulated.",
                 "Transparency rules often require telling people when they interact with AI or see AI-generated content.",
                 "Many countries and organisations publish AI principles covering fairness, accountability and transparency."),
            ),
            Lesson(
                "human-oversight", "Human oversight and accountability",
                ("Explain why humans stay accountable", "Design appropriate oversight"),
                ("People and organisations remain accountable for decisions made with the help of AI.",
                 "Human-in-the-loop designs have a person review or approve AI outputs before they take effect.",
                 "Automation bias is the tendency to trust automated outputs too readily, even when they are wrong.",
                 "The level of oversight should match the stakes: higher-impact decisions need more human review."),
            ),
        ),
    ),
    LearningPath(
        id="ai-for-business",
        title="AI for Business",
        description="Find the right problems, build or buy wisely, and measure whether AI is paying off.",
        level="beginner",
        topic="AI for Business",
        lessons=(
            Lesson(
                "finding-problems", "Finding problems worth solving",
                ("Spot good AI use cases", "Avoid solutions looking for problems"),
                ("Strong AI use cases start from a frequent, costly problem rather than from a technology.",
                 "Tasks with lots of examples, clear success criteria and tolerance for occasional errors suit AI well.",
                 "Talking to the people who do the work today reveals where time and money are really lost.",
                 "A simple rule or spreadsheet is sometimes a better solution than AI."),
            ),
            Lesson(
                "data-readiness", "Data readiness",
                ("Assess whether data supports a use case", "Plan to improve data"),
                ("AI projects depend on having relevant, accessible and good-quality data.",
                 "Data that is scattered across systems or inconsistently recorded slows projects down.",
                 "Clear ownership and permission to use data are as important as the data itself.",
                 "Starting with a small, well-defined dataset helps prove value before scaling."),
            ),
            Lesson(
                "build-vs-buy", "Build, buy or use an API",
                ("Compare build, buy and API options", "Weigh cost, control and speed"),
                ("Off-the-shelf AI products are fastest to adopt but offer less customisation.",
                 "Foundation model APIs let teams build custom applications without training models from scratch.",
                 "Building or fine-tuning your own models gives most control but needs specialised skills, data and budget.",
                 "Total cost includes integration, maintenance, monitoring and staff time, not just licence fees."),
            ),
            Lesson(
                "pilots-roi", "Pilots and measuring value",
                ("Design a useful pilot", "Measure return on investment"),
                ("A pilot tests an AI solution on a limited scope with clear success metrics agreed in advance.",
                 "Useful metrics compare the new process with the baseline, for example time saved, error rates or revenue.",
                 "Adoption matters: a tool that people do not use delivers no value.",
                 "Return on investment compares the benefits with the full cost of building and running the solution."),
            ),
            Lesson(
                "deploy-monitor", "Deploying and monitoring",
                ("Plan a responsible rollout", "Explain drift and monitoring"),
                ("Rolling out gradually lets teams catch problems before they affect everyone.",
                 "Data drift happens when real-world inputs change over time, which can degrade model performance.",
                 "Monitoring tracks quality, cost and user feedback after launch.",
                 "Clear fallback plans and human escalation paths keep the business running when the AI gets it wrong."),
            ),
        ),
    ),
)

PATHS_BY_ID = {p.id: p for p in PATHS}
LESSONS_BY_ID: dict[str, tuple[LearningPath, Lesson]] = {l.id: (p, l) for p in PATHS for l in p.lessons}


def find_lesson(lesson_id: str) -> tuple[LearningPath, Lesson] | None:
    return LESSONS_BY_ID.get(lesson_id)


def lesson_brief(lesson: Lesson) -> str:
    return (
        f"Lesson: {lesson.title}\nObjectives:\n" + "\n".join(f"- {o}" for o in lesson.objectives)
        + "\nKey points (accurate core; teach all of them):\n" + "\n".join(f"- {k}" for k in lesson.key_points)
    )


def path_from_dict(path_id: str, data: dict) -> LearningPath:
    """Build a learner's custom path (stored as JSON) with stable lesson ids."""
    lessons = tuple(
        Lesson(f"{path_id}-{i + 1}", l["title"], tuple(l["objectives"]), tuple(l["key_points"]), int(l.get("minutes", 10)))
        for i, l in enumerate(data["lessons"])
    )
    return LearningPath(path_id, data["title"], data["description"], data["level"], data.get("topic", ""), lessons)

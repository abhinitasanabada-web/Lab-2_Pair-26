"""Task 1.3 - per-conversation memory.

Each conversation_id gets its own ConversationMemory object. User-supplied values ("assume X for
this chat") live ONLY here, as plain text in the history - never in the vector index - and the
object is discarded when the conversation ends, so nothing can leak into another conversation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import labels_in_question


@dataclass
class Turn:
    question: str
    answer: dict  # the final JSON answer for that turn


@dataclass
class ConversationMemory:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)

    def add(self, question: str, answer: dict) -> None:
        self.turns.append(Turn(question, answer))

    def previous_questions(self, n: int = 2) -> list[str]:
        """Earlier user messages of THIS conversation (used to resolve follow-ups like 'And in 2013?')."""
        return [t.question for t in self.turns[-n:]]

    def previous_labels(self) -> list[str]:
        """Figure/table labels mentioned earlier in this conversation, most recent first."""
        out: list[str] = []
        for t in reversed(self.turns):
            for lab in labels_in_question(t.question):
                if lab not in out:
                    out.append(lab)
        return out

    def history_text(self, max_answer_chars: int = 500) -> str:
        """Render earlier turns for the prompt. Raw user messages are kept verbatim so any value the
        user supplied is visible to the model for this conversation only."""
        if not self.turns:
            return "(this is the first message of the conversation)"
        lines = []
        for i, t in enumerate(self.turns, 1):
            lines.append(f"User (turn {i}): {t.question}")
            lines.append(f"Assistant (turn {i}, status={t.answer.get('status', '')}): "
                         f"{str(t.answer.get('answer', ''))[:max_answer_chars]}")
        return "\n".join(lines)

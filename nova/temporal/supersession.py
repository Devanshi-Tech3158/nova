"""Temporal supersession logic for memories."""

from datetime import datetime

from nova.core.protocols import MemoryStore
from nova.core.schemas import Memory, MemoryStatus, MemoryType

SUPERSEDABLE_TYPES = {
    MemoryType.DECISION,
    MemoryType.CORRECTION,
    MemoryType.PREFERENCE,
}


def _link_out_of_order(
    store: MemoryStore,
    older_mem: Memory,
    current_mem: Memory,
) -> None:
    """Link an older superseded memory into the supersession chain."""
    if current_mem.supersedes_id is None:
        store.supersede(older_mem.id, current_mem.id)
        return

    curr_id: str | None = current_mem.supersedes_id
    parent = current_mem
    while curr_id:
        prev = store.get(curr_id)
        if prev is None:
            break
        if older_mem.timestamp > prev.timestamp:
            store.supersede(prev.id, older_mem.id)
            store.supersede(older_mem.id, parent.id)
            return
        parent = prev
        curr_id = prev.supersedes_id

    store.supersede(older_mem.id, parent.id)


def _apply_internal(store: MemoryStore, new_memories: list[Memory]) -> int:
    """Execute supersession logic over sorted incoming memories."""
    sorted_memories = sorted(new_memories, key=lambda m: (m.timestamp, m.id))
    superseded_count = 0

    for mem in sorted_memories:
        existing = store.get(mem.id)
        if existing is not None:
            if mem.type in SUPERSEDABLE_TYPES and hasattr(store, "conn") and store.conn is not None:
                cur = store.conn.cursor()
                rows = cur.execute(
                    "SELECT id, timestamp, type FROM memories WHERE subject = ? AND status = 'current' AND id != ?;",
                    (mem.subject, mem.id),
                ).fetchall()
                for r in rows:
                    r_ts = datetime.fromisoformat(r["timestamp"])
                    if r["type"] in [t.value for t in SUPERSEDABLE_TYPES] and mem.timestamp > r_ts:
                        store.supersede(r["id"], mem.id)
                        superseded_count += 1
            continue

        if mem.type not in SUPERSEDABLE_TYPES:
            store.add(mem)
            continue

        current = store.current_by_subject(mem.subject)
        if current is None or current.type not in SUPERSEDABLE_TYPES:
            store.add(mem)
            continue

        if mem.timestamp > current.timestamp:
            store.add(mem)
            store.supersede(current.id, mem.id)
            superseded_count += 1
        elif mem.timestamp < current.timestamp:
            mem_superseded = mem.model_copy(update={"status": MemoryStatus.SUPERSEDED})
            store.add(mem_superseded)
            _link_out_of_order(store, mem_superseded, current)
            superseded_count += 1
        else:
            if mem.id > current.id:
                store.add(mem)
                store.supersede(current.id, mem.id)
                superseded_count += 1
            else:
                mem_superseded = mem.model_copy(update={"status": MemoryStatus.SUPERSEDED})
                store.add(mem_superseded)
                _link_out_of_order(store, mem_superseded, current)
                superseded_count += 1

    return superseded_count


def apply(store: MemoryStore, new_memories: list[Memory]) -> int:
    """Apply temporal supersession rules to new memories in a single transaction."""
    if not new_memories:
        return 0

    if hasattr(store, "conn") and store.conn is not None:
        with store.conn:
            return _apply_internal(store, new_memories)
    return _apply_internal(store, new_memories)

"""
OASIS Agent Profile Generator

Converts Zep knowledge-graph entities into OasisAgentProfile instances
that the OASIS simulation engine can consume.

Improvements over the original:
1. Enriches entity context via Zep graph search before LLM generation.
2. Generates rich, detailed persona descriptions for each agent.
3. Distinguishes individual entities from group/institutional entities.
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from ..config import Config
from ..utils.logger import get_logger
from .memory import get_memory_backend
from .memory.base import EntityNode
from .entity_reader import EntityReader as ZepEntityReader

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """OASIS agent profile data structure."""

    # Core fields
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Reddit-style engagement
    karma: int = 1000

    # Twitter-style engagement
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Extended persona attributes
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Traceability back to the source entity
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """Return profile in Reddit OASIS format (JSON-serialisable dict)."""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS requires "username" (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        """Return profile in Twitter OASIS format (JSON-serialisable dict)."""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS requires "username" (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Return the full profile as a plain dict."""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """Converts Zep knowledge-graph entities into OasisAgentProfile instances.

    Features:
    1. Enriches each entity with Zep graph search results before LLM generation.
    2. Generates detailed personas covering background, personality, and behaviour.
    3. Distinguishes individual entities from group/institutional entities.
    """

    # All 16 MBTI types
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # Common countries for default profile generation
    COUNTRIES = [
        "China", "US", "UK", "Japan", "Germany", "France", 
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]
    
    # Individual entity types (generate a concrete personal persona)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure", 
        "expert", "faculty", "official", "journalist", "activist"
    ]
    
    # Group/institutional entity types (generate a representative account persona)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo", 
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        graph_id: Optional[str] = None,
        **_kwargs: Any,  # absorb legacy keyword args (e.g. legacy api_key kwargs)
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # Memory backend used to enrich entity context via graph search
        self._backend = None
        self.graph_id = graph_id

        try:
            self._backend = get_memory_backend()
        except Exception as exc:
            logger.warning(f"Memory backend initialisation failed: {exc}")
    
    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """Generate an OasisAgentProfile for a single Zep entity node.

        Args:
            entity:   Zep entity node.
            user_id:  Integer ID for the OASIS agent.
            use_llm:  If True, use the LLM to generate a detailed persona;
                      otherwise fall back to rule-based generation.

        Returns:
            OasisAgentProfile instance.
        """
        entity_type = entity.get_entity_type() or "Entity"

        name = entity.name
        user_name = self._generate_username(name)

        context = self._build_entity_context(entity)

        if use_llm:
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Derive a unique username from *name* (lowercase, underscores, random suffix)."""
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_graph_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Search the memory backend for context related to an entity.

        Returns a dict with keys: facts (list[str]), node_summaries (list[str]),
        context (str).
        """
        if not self._backend:
            return {"facts": [], "node_summaries": [], "context": ""}

        if not self.graph_id:
            logger.debug("Skipping graph search: graph_id not set")
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name
        query = f"All information, activities, events, relationships and background about {entity_name}"

        results: Dict[str, Any] = {"facts": [], "node_summaries": [], "context": ""}

        try:
            edge_results = self._backend.search_edges(self.graph_id, query, limit=30)
            for edge in edge_results.edges:
                if edge.fact:
                    results["facts"].append(edge.fact)

            node_results = self._backend.search_nodes(self.graph_id, query, limit=20)
            for node in node_results.nodes:
                if node.summary:
                    results["node_summaries"].append(f"[{node.name}]: {node.summary}")
                    results["facts"].append(f"[{node.name}]: {node.summary}")

            logger.info(f"Graph search complete: {len(results['facts'])} relevant facts")

        except NotImplementedError:
            logger.debug("Memory backend search not implemented — skipping entity enrichment")
        except Exception as exc:
            logger.warning(f"Graph search failed for entity '{entity_name}': {exc}")

        return results

    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """Build a rich context string for *entity* by combining graph data and search results.

        Sections included:
        1. Entity attributes
        2. Related edges (facts/relationships)
        3. Related node summaries
        4. Zep search results
        """
        context_parts = []

        # 1. Entity attributes
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Entity Attributes\n" + "\n".join(attrs))

        # 2. Related edges (facts and relationships)
        existing_facts: set = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (related entity)")
                    else:
                        relationships.append(f"- (related entity) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### Related Facts and Relationships\n" + "\n".join(relationships))

        # 3. Related node summaries
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Related Entity Information\n" + "\n".join(related_info))

        # 4. Zep search enrichment
        graph_results = self._search_graph_for_entity(entity)

        if graph_results.get("facts"):
            new_facts = [f for f in graph_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append(
                    "### Facts Retrieved from Zep\n"
                    + "\n".join(f"- {f}" for f in new_facts[:15])
                )

        if graph_results.get("node_summaries"):
            context_parts.append(
                "### Related Nodes Retrieved from Zep\n"
                + "\n".join(f"- {s}" for s in graph_results["node_summaries"][:10])
            )

        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Return True if *entity_type* represents an individual person."""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Return True if *entity_type* represents a group or institution."""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """Use the LLM to generate a detailed persona for an entity.

        Individual entities get a concrete personal persona; group/institutional
        entities get a representative account persona. Up to 3 attempts are made
        with decreasing temperature; falls back to rule-based generation on failure.
        """
        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # lower temperature on each retry
                    # max_tokens intentionally not set — let the LLM decide
                )

                content = response.choices[0].message.content

                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning("LLM output truncated (attempt %d), attempting repair...", attempt + 1)
                    content = self._fix_truncated_json(content)

                try:
                    result = json.loads(content)

                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} is a {entity_type}."

                    return result

                except json.JSONDecodeError as je:
                    logger.warning("JSON parse failed (attempt %d): %s", attempt + 1, str(je)[:80])

                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning("LLM call failed (attempt %d): %s", attempt + 1, str(e)[:80])
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # exponential back-off

        logger.warning(
            "LLM persona generation failed after %d attempts: %s — using rule-based fallback",
            max_attempts, last_error,
        )
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Attempt to close a JSON object that was cut short by the model."""
        import re

        content = content.strip()

        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Close an unclosed string value
        if content and content[-1] not in '",}]':
            content += '"'

        content += ']' * open_brackets
        content += '}' * open_braces

        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Attempt several strategies to recover a parseable JSON object from *content*."""
        import re

        # Strategy 1: close truncated structure
        content = self._fix_truncated_json(content)

        # Strategy 2: extract the outermost {...} block and normalise newlines
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            def fix_string_newlines(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)

            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError:
                # Strategy 3: strip control characters
                try:
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except Exception:
                    pass

        # Strategy 4: extract partial fields with regex
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # may be truncated

        bio = bio_match.group(1) if bio_match else (
            entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
        )
        persona = persona_match.group(1) if persona_match else (
            entity_summary or f"{entity_name} is a {entity_type}."
        )

        if bio_match or persona_match:
            logger.info("Extracted partial fields from malformed JSON response")
            return {"bio": bio, "persona": persona, "_fixed": True}

        # Strategy 5: give up and return a minimal fallback
        logger.warning("JSON repair failed — returning minimal fallback structure")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} is a {entity_type}.",
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """Return the system prompt for profile generation."""
        return (
            "You are a social-media persona designer for ManaSim, a social simulation engine. "
            "Generate detailed, realistic agent profiles for opinion-dynamics simulation. "
            "Your output must be valid JSON. All string values must not contain unescaped newline characters. "
            "Write in English."
        )
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build the LLM prompt for generating a persona for an individual entity."""
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "none"
        context_str = context[:3000] if context else "No additional context available."

        return f"""Generate a detailed social-media persona for the following individual entity.
Reproduce real-world details as accurately as possible based on the provided information.

Entity name   : {entity_name}
Entity type   : {entity_type}
Entity summary: {entity_summary}
Entity attrs  : {attrs_str}

Context:
{context_str}

Return a JSON object with exactly these fields:

1. bio         : Social-media profile bio (up to 200 words).
2. persona     : Detailed persona description (plain text, ~500 words). Include:
                 - Background (age, profession, education, location)
                 - Personal history (key experiences, connection to events, social relationships)
                 - Personality (MBTI type, core traits, emotional expression)
                 - Online behaviour (posting frequency, content preferences, interaction style)
                 - Stance & opinions (attitude toward the domain topic; triggers)
                 - Distinctive traits (catchphrases, hobbies, memorable experiences)
                 - Personal memories (how this individual relates to the simulation event)
3. age         : Integer age.
4. gender      : "male" or "female".
5. mbti        : One of the 16 MBTI types (e.g. INTJ, ENFP).
6. country     : Country name in English.
7. profession  : Job title or occupation.
8. interested_topics : JSON array of topic strings.

Rules:
- All values must be strings or numbers — no unescaped newlines.
- persona must be a single continuous block of text.
- age must be a valid integer; gender must be "male" or "female".
- All content must be consistent with the provided entity information.
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Build the LLM prompt for generating a persona for a group or institutional entity."""
        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "none"
        context_str = context[:3000] if context else "No additional context available."

        return f"""Generate a detailed social-media account profile for the following group or institutional entity.
Reproduce real-world details as accurately as possible based on the provided information.

Entity name   : {entity_name}
Entity type   : {entity_type}
Entity summary: {entity_summary}
Entity attrs  : {attrs_str}

Context:
{context_str}

Return a JSON object with exactly these fields:

1. bio         : Official account bio (up to 200 words, professional tone).
2. persona     : Detailed account profile description (plain text, ~500 words). Include:
                 - Institutional background (full name, type, founding context, core functions)
                 - Account positioning (account type, target audience, key purposes)
                 - Communication style (language register, typical phrases, off-limits topics)
                 - Content characteristics (content types, posting frequency, active periods)
                 - Stance & policy (official position on the domain topic; approach to controversy)
                 - Institutional memory (how this entity relates to the simulation event)
3. age         : Use 30 (conventional placeholder for institutional accounts).
4. gender      : Use "other" (institutional accounts are not individuals).
5. mbti        : MBTI type reflecting the account's communication style (e.g. ISTJ for formal/conservative).
6. country     : Country name in English.
7. profession  : Description of the institution's function.
8. interested_topics : JSON array of focus-area strings.

Rules:
- All values must be strings or numbers — no null values, no unescaped newlines.
- persona must be a single continuous block of text.
- age must be the integer 30; gender must be the string "other".
- Statements made by this account must be consistent with the institution's identity.
"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a minimal rule-based profile when LLM generation is unavailable."""
        entity_type_lower = entity_type.lower()
        
        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }
        
        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }
        
        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,        # conventional placeholder for institutional accounts
                "gender": "other",
                "mbti": "ISTJ",   # formal and methodical communication style
                "country": "Unknown",
                "profession": "Media",
                "interested_topics": ["General News", "Current Events", "Public Affairs"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,        # conventional placeholder for institutional accounts
                "gender": "other",
                "mbti": "ISTJ",   # formal and methodical communication style
                "country": "Unknown",
                "profession": entity_type,
                "interested_topics": ["Public Policy", "Community", "Official Announcements"],
            }

        else:
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Set the graph ID used for Zep context enrichment."""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """Generate OasisAgentProfile instances for a list of entities (parallel).

        Args:
            entities:            List of EntityNode objects.
            use_llm:             If True, use LLM for persona generation.
            progress_callback:   Callback(current, total, message) for progress updates.
            graph_id:            Zep graph ID for context enrichment.
            parallel_count:      Max concurrent LLM calls (default 5).
            realtime_output_path: If set, write profiles to disk after each completion.
            output_platform:     "reddit" or "twitter" — controls realtime save format.

        Returns:
            List of OasisAgentProfile instances in entity order.
        """
        import concurrent.futures
        from threading import Lock

        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # pre-allocated to preserve order
        completed_count = [0]      # mutable counter for closure access
        lock = Lock()

        def save_profiles_realtime():
            """Write all completed profiles to disk."""
            if not realtime_output_path:
                return

            with lock:
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning("Real-time profile save failed: %s", e)

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Worker function — generate one profile."""
            entity_type = entity.get_entity_type() or "Entity"
            
            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )
                
                self._print_generated_profile(entity.name, entity_type, profile)
                
                return idx, profile, None
                
            except Exception as e:
                logger.error("Failed to generate profile for entity '%s': %s", entity.name, e)
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)
        
        logger.info("Starting parallel profile generation: %d entities, %d workers", total, parallel_count)
        print(f"\n{'='*60}")
        print(f"Generating agent profiles — {total} entities, {parallel_count} parallel workers")
        print(f"{'='*60}\n")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    save_profiles_realtime()
                    
                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completed {current}/{total}: {entity.name} ({entity_type})"
                        )

                    if error:
                        logger.warning("[%d/%d] %s used fallback profile: %s", current, total, entity.name, error)
                    else:
                        logger.info("[%d/%d] Profile generated: %s (%s)", current, total, entity.name, entity_type)

                except Exception as e:
                    logger.error("Exception while processing entity '%s': %s", entity.name, e)
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    save_profiles_realtime()

        print(f"\n{'='*60}")
        print(f"Profile generation complete — {len([p for p in profiles if p])} agents generated.")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Print a newly generated profile to stdout (full content, no truncation)."""
        separator = "-" * 70

        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'none'

        output_lines = [
            f"\n{separator}",
            f"[Generated] {entity_name} ({entity_type})",
            f"{separator}",
            f"Username: {profile.user_name}",
            f"",
            f"[Bio]",
            f"{profile.bio}",
            f"",
            f"[Persona]",
            f"{profile.persona}",
            f"",
            f"[Attributes]",
            f"Age: {profile.age} | Gender: {profile.gender} | MBTI: {profile.mbti}",
            f"Profession: {profile.profession} | Country: {profile.country}",
            f"Topics: {topics_str}",
            separator
        ]
        
        output = "\n".join(output_lines)
        
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """Save profiles to disk in the format required by the OASIS platform.

        Platform → file format mapping:
        - "twitter"      → CSV  (generate_twitter_agent_graph expects CSV)
        - "reddit"       → JSON
        - "classroom"    → JSON (classroom uses Reddit OASIS platform)
        - "organisation" → JSON (organisation script converts JSON → CSV at runtime)

        Args:
            profiles:  List of OasisAgentProfile instances.
            file_path: Destination file path.
            platform:  One of "reddit", "twitter", "classroom", "organisation".
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            # reddit, classroom, and organisation all use the JSON format
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """Save Twitter profiles as CSV (required by generate_twitter_agent_graph).

        OASIS Twitter CSV columns:
        - user_id     : sequential integer ID
        - name        : display name
        - username    : account handle
        - user_char   : full persona injected into the LLM system prompt
        - description : short public bio shown on the profile page

        user_char vs description:
        - user_char   : internal; controls how the agent thinks and acts
        - description : external; visible to other agents
        """
        import csv

        # Ensure .csv extension
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            writer.writerow(['user_id', 'name', 'username', 'user_char', 'description'])

            for idx, profile in enumerate(profiles):
                # user_char: full persona (bio + persona) for the LLM system prompt
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')

                description = profile.bio.replace('\n', ' ').replace('\r', ' ')

                writer.writerow([
                    idx,              # user_id (0-based sequential)
                    profile.name,
                    profile.user_name,
                    user_char,
                    description,
                ])

        logger.info("Saved %d Twitter profiles to %s (OASIS CSV format)", len(profiles), file_path)
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """Normalise a gender string to the values OASIS accepts: male, female, other."""
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        gender_map = {
            # English
            "male": "male",
            "female": "female",
            "other": "other",
            # Legacy Chinese values kept for backwards compatibility with old profiles
            "\u7537": "male",    # Chinese: male
            "\u5973": "female",  # Chinese: female
            "\u673a\u6784": "other",  # Chinese: institution
            "\u5176\u4ed6": "other",  # Chinese: other
        }

        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """Save profiles as JSON (used by Reddit/classroom/organisation environments).

        The format matches to_reddit_format() so OASIS can load it correctly.
        user_id is mandatory — it is the key OASIS uses in agent_graph.get_agent().

        Required fields:
        - user_id   : integer; must match poster_agent_id in initial_posts
        - username  : account handle
        - name      : display name
        - bio       : short public bio
        - persona   : detailed behavioural description
        - age       : integer
        - gender    : "male", "female", or "other"
        - mbti      : MBTI type string
        - country   : country name
        """
        data = []
        for idx, profile in enumerate(profiles):
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else profile.name,
                "persona": (
                    profile.persona
                    or f"{profile.name} is a participant in social discussions."
                ),
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Required by OASIS — always include with defaults
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "Unknown",
            }

            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Saved %d profiles to %s (JSON format, user_id included)",
            len(profiles), file_path,
        )
    
    # Deprecated alias kept for backwards compatibility
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Deprecated] Use save_profiles() instead."""
        logger.warning("save_profiles_to_json is deprecated — use save_profiles()")
        self.save_profiles(profiles, file_path, platform)


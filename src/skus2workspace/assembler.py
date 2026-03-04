"""Step 1: Assemble workspace by copying and reorganizing SKUs."""

import json
import re
import shutil
from pathlib import Path

import structlog

from skus2workspace.schemas.workspace import WorkspaceManifest
from chunks2skus.schemas.sku import Glossary, LabelTree, Relationships

logger = structlog.get_logger(__name__)

# SKU subdirectories to copy
SKU_SUBDIRS = ["factual", "procedural", "relational"]

# Regex to match any prefix before a known SKU subdirectory
# e.g. "test_data/basel_skus/factual/sku_001" → "skus/factual/sku_001"
# e.g. "output/skus/procedural/skill_003" → "skus/procedural/skill_003"
# e.g. "test_data/basel_skus/meta" → "skus/meta" (no trailing slash)
PATH_REWRITE_PATTERN = re.compile(
    r"(?:^|(?<=[\s(/\"']))[\w./\-]+?(?=(?:factual|procedural|relational|meta)(?:/|$|\s|\"|\)|,))"
)


def _rewrite_path(text: str) -> tuple[str, int]:
    """
    Rewrite SKU paths in text, replacing any prefix before
    factual/procedural/relational/meta/ with 'skus/'.

    Returns:
        Tuple of (rewritten text, number of replacements).
    """
    count = 0

    def replacer(match: re.Match) -> str:
        nonlocal count
        count += 1
        return "skus/"

    result = PATH_REWRITE_PATTERN.sub(replacer, text)
    return result, count


class WorkspaceAssembler:
    """Copies and reorganizes SKUs into a self-contained workspace."""

    def __init__(self, skus_dir: Path, workspace_dir: Path, merge: bool = False):
        self.skus_dir = Path(skus_dir).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.merge = merge

    def assemble(self) -> WorkspaceManifest:
        """
        Run the full assembly process.

        Returns:
            WorkspaceManifest with counts and status.
        """
        logger.info(
            "Starting workspace assembly",
            skus_dir=str(self.skus_dir),
            workspace_dir=str(self.workspace_dir),
            merge=self.merge,
        )

        manifest = WorkspaceManifest(
            source_skus_dir=str(self.skus_dir),
            workspace_dir=str(self.workspace_dir),
        )

        # Validate source
        if not self.skus_dir.exists():
            logger.error("SKUs directory does not exist", path=str(self.skus_dir))
            raise FileNotFoundError(f"SKUs directory not found: {self.skus_dir}")

        mapping_path = self.skus_dir / "meta" / "mapping.md"
        if not mapping_path.exists():
            logger.warning("mapping.md not found in meta/", path=str(mapping_path))

        # Create workspace
        skus_dest = self.workspace_dir / "skus"
        skus_dest.mkdir(parents=True, exist_ok=True)

        total_files = 0
        self.renamed_skus_map = {}

        # 1. Copy/Merge SKU subdirectories
        for subdir in SKU_SUBDIRS:
            src = self.skus_dir / subdir
            dst = skus_dest / subdir
            
            if src.exists():
                if dst.exists():
                    if self.merge:
                        if subdir == "relational":
                            count = self._merge_relational(src, dst)
                            total_files += count
                        else:
                            # Factual/Procedural
                            count = self._merge_sku_dir(src, dst, subdir)
                            total_files += count
                    else:
                        shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
                        total_files += file_count
                        logger.info("Copied directory", subdir=subdir, files=file_count)
                else:
                    shutil.copytree(src, dst)
                    file_count = sum(1 for _ in dst.rglob("*") if _.is_file())
                    total_files += file_count
                    logger.info("Copied directory", subdir=subdir, files=file_count)

        # Count SKUs
        factual_dir = skus_dest / "factual"
        procedural_dir = skus_dest / "procedural"
        if factual_dir.exists():
            manifest.factual_count = sum(1 for d in factual_dir.iterdir() if d.is_dir())
        if procedural_dir.exists():
            manifest.procedural_count = sum(1 for d in procedural_dir.iterdir() if d.is_dir())
        manifest.has_relational = (skus_dest / "relational").exists()

        # 2. Copy/Merge postprocessing/ if exists
        postproc_src = self.skus_dir / "postprocessing"
        if postproc_src.exists():
            postproc_dst = skus_dest / "postprocessing"
            if postproc_dst.exists():
                if not self.merge:
                    shutil.rmtree(postproc_dst)
                # Overwrite/Copy postprocessing
                if not self.merge or not postproc_dst.exists():
                    shutil.copytree(postproc_src, postproc_dst)
                else:
                    # If merge and exists, we overwrite for simplicity
                    shutil.rmtree(postproc_dst)
                    shutil.copytree(postproc_src, postproc_dst)
            else:
                shutil.copytree(postproc_src, postproc_dst)
            
            pp_count = sum(1 for _ in postproc_dst.rglob("*") if _.is_file())
            total_files += pp_count
            logger.info("Copied postprocessing", files=pp_count)

        # 3. Copy/Merge skus_index.json
        index_src = self.skus_dir / "skus_index.json"
        index_dst = skus_dest / "skus_index.json"
        
        if index_src.exists():
            if self.merge and index_dst.exists():
                rewrite_count = self._merge_skus_index(index_src, index_dst)
                manifest.paths_rewritten += rewrite_count
                total_files += 1
                logger.info("Merged skus_index.json", paths_rewritten=rewrite_count)
            else:
                rewrite_count = self._rewrite_skus_index(index_src, index_dst)
                manifest.paths_rewritten += rewrite_count
                total_files += 1
                logger.info("Copied and rewrote skus_index.json", paths_rewritten=rewrite_count)

        # 4. Copy/Merge eureka.md
        eureka_src = self.skus_dir / "meta" / "eureka.md"
        eureka_dst = self.workspace_dir / "eureka.md"
        
        if eureka_src.exists():
            if self.merge and eureka_dst.exists():
                self._append_file(eureka_src, eureka_dst, "\n\n## New Insights\n\n")
                logger.info("Appended eureka.md")
            else:
                shutil.copy2(eureka_src, eureka_dst)
                logger.info("Copied eureka.md to workspace root")
            manifest.has_eureka = True
            total_files += 1

        # 5. Rewrite/Merge mapping.md
        mapping_dst = self.workspace_dir / "mapping.md"
        
        if mapping_path.exists():
            content = mapping_path.read_text(encoding="utf-8")
            rewritten, rewrite_count = _rewrite_path(content)
            
            if self.merge and mapping_dst.exists():
                # Append content
                existing_content = mapping_dst.read_text(encoding="utf-8")
                merged_content = existing_content + "\n\n" + rewritten
                mapping_dst.write_text(merged_content, encoding="utf-8")
                logger.info("Merged mapping.md", paths_rewritten=rewrite_count)
            else:
                mapping_dst.write_text(rewritten, encoding="utf-8")
                logger.info("Rewrote mapping.md", paths_rewritten=rewrite_count)
                
            manifest.has_mapping = True
            manifest.paths_rewritten += rewrite_count
            total_files += 1

        manifest.total_files_copied = total_files

        logger.info(
            "Assembly complete",
            total_files=total_files,
            factual=manifest.factual_count,
            procedural=manifest.procedural_count,
            paths_rewritten=manifest.paths_rewritten,
        )

        return manifest

    def _merge_sku_dir(self, src_dir: Path, dst_dir: Path, subdir: str) -> int:
        """Merge SKU directory by copying and renumbering conflicts."""
        # Get max ID in destination
        max_id = 0
        existing_skus = list(dst_dir.glob("*"))
        
        prefix_map = {
            "factual": "sku_",
            "procedural": "skill_"
        }
        prefix = prefix_map.get(subdir, "item_")
        
        for p in existing_skus:
            if p.is_dir() and p.name.startswith(prefix):
                try:
                    num = int(p.name.split("_")[-1])
                    max_id = max(max_id, num)
                except ValueError:
                    pass
        
        count = 0
        for src_sku in src_dir.iterdir():
            if not src_sku.is_dir():
                continue
                
            # Check for conflict
            dst_sku = dst_dir / src_sku.name
            
            if dst_sku.exists():
                # Renumber
                max_id += 1
                new_name = f"{prefix}{max_id:03d}"
                new_dst = dst_dir / new_name
                shutil.copytree(src_sku, new_dst)
                
                self.renamed_skus_map[src_sku.name] = new_name
                count += sum(1 for _ in new_dst.rglob("*") if _.is_file())
            else:
                shutil.copytree(src_sku, dst_sku)
                count += sum(1 for _ in dst_sku.rglob("*") if _.is_file())
                
        return count

    def _merge_relational(self, src_dir: Path, dst_dir: Path) -> int:
        """Merge relational JSON files."""
        files = ["label_tree.json", "glossary.json", "relationships.json"]
        count = 0
        
        for fname in files:
            src_f = src_dir / fname
            dst_f = dst_dir / fname
            
            if src_f.exists():
                if dst_f.exists():
                    try:
                        if fname == "label_tree.json":
                            self._merge_label_tree_files(src_f, dst_f)
                        elif fname == "glossary.json":
                            self._merge_glossary_files(src_f, dst_f)
                        elif fname == "relationships.json":
                            self._merge_relationships_files(src_f, dst_f)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to merge {fname}", error=str(e))
                else:
                    shutil.copy2(src_f, dst_f)
                    count += 1
        
        # Merge header.md if exists
        header_src = src_dir / "header.md"
        header_dst = dst_dir / "header.md"
        if header_src.exists() and not header_dst.exists():
             shutil.copy2(header_src, header_dst)
             count += 1
             
        return count

    def _merge_label_tree_files(self, src: Path, dst: Path) -> None:
        src_data = LabelTree.model_validate_json(src.read_text(encoding="utf-8"))
        dst_data = LabelTree.model_validate_json(dst.read_text(encoding="utf-8"))
        
        for root in src_data.roots:
            self._merge_label_node(dst_data.roots, root)
            
        dst.write_text(dst_data.model_dump_json(indent=2), encoding="utf-8")

    def _merge_label_node(self, existing_list: list, new_node) -> None:
        existing = None
        for node in existing_list:
            if node.name.lower() == new_node.name.lower():
                existing = node
                break
        
        if existing:
            for child in new_node.children:
                self._merge_label_node(existing.children, child)
        else:
            existing_list.append(new_node)

    def _merge_glossary_files(self, src: Path, dst: Path) -> None:
        src_data = Glossary.model_validate_json(src.read_text(encoding="utf-8"))
        dst_data = Glossary.model_validate_json(dst.read_text(encoding="utf-8"))
        
        for entry in src_data.entries:
            dst_data.add_or_update(entry)
            
        dst.write_text(dst_data.model_dump_json(indent=2), encoding="utf-8")

    def _merge_relationships_files(self, src: Path, dst: Path) -> None:
        src_data = Relationships.model_validate_json(src.read_text(encoding="utf-8"))
        dst_data = Relationships.model_validate_json(dst.read_text(encoding="utf-8"))
        
        for rel in src_data.entries:
            dst_data.add(rel)
            
        dst.write_text(dst_data.model_dump_json(indent=2), encoding="utf-8")

    def _merge_skus_index(self, src: Path, dst: Path) -> int:
        src_data = json.loads(src.read_text(encoding="utf-8"))
        dst_data = json.loads(dst.read_text(encoding="utf-8"))
        
        count = 0
        dst_skus = dst_data.get("skus", [])
        
        for entry in src_data.get("skus", []):
            old_path = entry.get("path", "")
            folder_name = Path(old_path).name
            
            if folder_name in self.renamed_skus_map:
                new_folder_name = self.renamed_skus_map[folder_name]
                rewritten_path, _ = _rewrite_path(old_path)
                parent = str(Path(rewritten_path).parent).replace("\\", "/")
                entry["path"] = f"{parent}/{new_folder_name}"
                entry["sku_id"] = new_folder_name
                entry["name"] = entry["name"].replace(folder_name, new_folder_name)
                count += 1
            else:
                new_path, n = _rewrite_path(old_path)
                if n > 0:
                    entry["path"] = new_path
                    count += n
            
            dst_skus.append(entry)
            
        dst_data["skus"] = dst_skus
        dst.write_text(json.dumps(dst_data, indent=2, ensure_ascii=False), encoding="utf-8")
        return count

    def _append_file(self, src: Path, dst: Path, separator: str = "\n") -> None:
        content = src.read_text(encoding="utf-8")
        with open(dst, "a", encoding="utf-8") as f:
            f.write(separator)
            f.write(content)

    def _rewrite_skus_index(self, src: Path, dst: Path) -> int:
        """
        Load skus_index.json, rewrite path fields, save to dst.

        Returns:
            Number of paths rewritten.
        """
        data = json.loads(src.read_text(encoding="utf-8"))
        count = 0

        for entry in data.get("skus", []):
            old_path = entry.get("path", "")
            if old_path:
                new_path, n = _rewrite_path(old_path)
                if n > 0:
                    entry["path"] = new_path
                    count += n

        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return count

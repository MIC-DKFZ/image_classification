#!/usr/bin/env python3
from __future__ import annotations

import argparse
import netrc
import os
from typing import Iterable

import requests


WANDB_GRAPHQL_URL = "https://api.wandb.ai/graphql"
SKIPPABLE_WANDB_STATES = {"finished", "running"}
DEFAULT_PAGE_SIZE = 1000
DEFAULT_TIMEOUT_SECONDS = 60
RUNS_QUERY = """
query ProjectRunNames($entity: String!, $project: String!, $cursor: String, $pageSize: Int!) {
  project(name: $project, entityName: $entity) {
    runs(first: $pageSize, after: $cursor) {
      edges {
        node {
          id
          displayName
          state
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
""".strip()


def resolve_wandb_api_key() -> str:
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        return api_key

    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth and auth[2]:
        return auth[2]

    try:
        auth = netrc.netrc().authenticators("wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if auth and auth[2]:
        return auth[2]

    raise RuntimeError(
        "Could not resolve a W&B API key. Set WANDB_API_KEY or configure ~/.netrc."
    )


def get_skippable_run_names(
    entity: str,
    project: str,
    target_run_names: Iterable[str],
    *,
    api_key: str | None = None,
    page_size: int = DEFAULT_PAGE_SIZE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    verbose: bool = False,
) -> set[str]:
    api_key = api_key or resolve_wandb_api_key()
    pending_names = set(target_run_names)
    if not pending_names:
        return set()

    if verbose:
        print(
            f"Fetching W&B runs for {entity}/{project} via GraphQL "
            f"for {len(pending_names)} target run names...",
            flush=True,
        )

    found_names: set[str] = set()
    cursor: str | None = None
    page_index = 0

    while True:
        page_index += 1
        response = requests.post(
            WANDB_GRAPHQL_URL,
            auth=("api", api_key),
            json={
                "query": RUNS_QUERY,
                "variables": {
                    "entity": entity,
                    "project": project,
                    "cursor": cursor,
                    "pageSize": page_size,
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(f"W&B GraphQL query failed: {payload['errors']}")

        project_data = payload.get("data", {}).get("project")
        if project_data is None:
            raise RuntimeError(
                f"W&B GraphQL query returned no project data for {entity}/{project}."
            )

        runs_data = project_data["runs"]
        edges = runs_data["edges"]
        page_matches = 0
        for edge in edges:
            node = edge["node"]
            display_name = node.get("displayName")
            state = node.get("state")
            if display_name in pending_names and state in SKIPPABLE_WANDB_STATES:
                if display_name not in found_names:
                    found_names.add(display_name)
                    page_matches += 1

        if verbose:
            print(
                f"W&B GraphQL page {page_index}: fetched {len(edges)} runs, "
                f"matched {page_matches}, total matched {len(found_names)}.",
                flush=True,
            )

        if found_names == pending_names:
            break

        page_info = runs_data["pageInfo"]
        if not page_info["hasNextPage"]:
            break
        cursor = page_info["endCursor"]

    if verbose:
        print(
            f"W&B GraphQL lookup complete: found {len(found_names)} completed/running runs to skip.",
            flush=True,
        )

    return found_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query W&B GraphQL for run names that are finished or running."
    )
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", action="append", default=[])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    names = get_skippable_run_names(
        args.entity,
        args.project,
        args.run_name,
        verbose=args.verbose,
    )
    for name in sorted(names):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

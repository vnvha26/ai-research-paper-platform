CITATION_ANALYSIS_QUERY = """
MATCH (source:Paper {paper_id: $paper_id})
OPTIONAL MATCH (source)-[:CITES]->(target:Paper)
WITH source, target
WHERE target IS NOT NULL
OPTIONAL MATCH (source)-[:CITES]-(sn:Paper)
WHERE sn.paper_id <> source.paper_id AND sn.paper_id <> target.paper_id
WITH source, target, collect(DISTINCT sn.paper_id) AS source_neighbors
OPTIONAL MATCH (target)-[:CITES]-(tn:Paper)
WHERE tn.paper_id <> source.paper_id AND tn.paper_id <> target.paper_id
WITH source, target, source_neighbors,
     collect(DISTINCT tn.paper_id) AS target_neighbors,
     EXISTS((target)-[:CITES]->(source)) AS is_mutual
WITH source, target, is_mutual, source_neighbors, target_neighbors,
     [x IN source_neighbors WHERE x IN target_neighbors] AS common_neighbors,
     source_neighbors + [x IN target_neighbors WHERE NOT x IN source_neighbors] AS union_neighbors
WITH source, target, is_mutual, source_neighbors, target_neighbors,
     common_neighbors, union_neighbors,
     [source.paper_id, target.paper_id] + common_neighbors[0..50] AS local_nodes
OPTIONAL MATCH (a:Paper)-[:CITES]->(b:Paper)
WHERE a.paper_id IN local_nodes
  AND b.paper_id IN local_nodes
  AND a.paper_id <> b.paper_id
WITH source, target, is_mutual, source_neighbors, target_neighbors,
     common_neighbors, union_neighbors, local_nodes,
     count(DISTINCT toString(a.paper_id) + '->' + toString(b.paper_id)) AS local_edge_count
RETURN target.paper_id AS cited_id,
       target.title AS cited_title,
       is_mutual,
       size(source_neighbors) AS source_neighbor_count,
       size(target_neighbors) AS target_neighbor_count,
       size(common_neighbors) AS common_neighbor_count,
       size(union_neighbors) AS union_neighbor_count,
       common_neighbors[0..10] AS common_neighbor_sample,
       size(local_nodes) AS local_node_count,
       local_edge_count
"""


def fetch_citation_records(neo_driver, paper_id):
    with neo_driver.session() as session:
        result = session.run(CITATION_ANALYSIS_QUERY, paper_id=str(paper_id))
        return [record.data() for record in result]

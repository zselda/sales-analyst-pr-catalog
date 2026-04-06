"""
Interactive Network Graph Visualizer
=====================================
Generates a standalone, interactive HTML file from the network_mapper output.
Uses D3.js force-directed graph with:
- Draggable nodes
- Hover tooltips with account details
- Zoom & pan
- Color-coded nodes (customer/supplier/bank/target)
- Animated edges with flow direction
- Glassmorphic sidebar with stats
"""

import json
import logging
import os
import re

logger = logging.getLogger("swarm.graph_visualizer")


def generate_network_html(network_data: dict, company_name: str = "Company") -> str:
    """
    Generate an interactive D3.js force-directed graph HTML from network_mapper output.

    Args:
        network_data: dict with 'nodes', 'edges', 'stats' from NetworkMapperAgent
        company_name: Company name for the title

    Returns:
        Complete HTML string
    """
    nodes = network_data.get("nodes", [])
    edges = network_data.get("edges", [])
    stats = network_data.get("stats", {})

    # Serialize for JS
    nodes_json = json.dumps(nodes, ensure_ascii=False)
    edges_json = json.dumps(edges, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{company_name} — Commercial Network Graph</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: #0a0e27;
            overflow: hidden;
            color: #e0e0e0;
        }}

        #graph-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}

        svg {{
            width: 100%;
            height: 100%;
            cursor: grab;
        }}
        svg:active {{
            cursor: grabbing;
        }}

        /* Glassmorphic sidebar */
        .sidebar {{
            position: fixed;
            top: 20px;
            left: 20px;
            width: 320px;
            background: rgba(15, 20, 50, 0.85);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            z-index: 100;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }}

        .sidebar h1 {{
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 4px;
            letter-spacing: -0.3px;
        }}

        .sidebar .subtitle {{
            font-size: 12px;
            color: #8b8fa3;
            margin-bottom: 20px;
            font-weight: 400;
        }}

        .stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .stat-card {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 10px;
            padding: 12px;
            text-align: center;
        }}

        .stat-card .stat-value {{
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }}

        .stat-card .stat-label {{
            font-size: 10px;
            color: #8b8fa3;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 2px;
        }}

        .legend {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            color: #c0c4d6;
        }}

        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }}

        .legend-count {{
            margin-left: auto;
            font-weight: 600;
            color: #ffffff;
            font-size: 13px;
        }}

        /* Tooltip */
        .tooltip {{
            position: fixed;
            pointer-events: none;
            background: rgba(10, 14, 39, 0.95);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 12px;
            padding: 14px 18px;
            font-size: 13px;
            z-index: 200;
            opacity: 0;
            transition: opacity 0.15s ease;
            max-width: 300px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
        }}

        .tooltip.visible {{
            opacity: 1;
        }}

        .tooltip .tt-label {{
            font-weight: 600;
            font-size: 14px;
            color: #ffffff;
            margin-bottom: 6px;
        }}

        .tooltip .tt-row {{
            display: flex;
            justify-content: space-between;
            gap: 16px;
            margin-top: 3px;
        }}

        .tooltip .tt-key {{
            color: #8b8fa3;
        }}

        .tooltip .tt-val {{
            color: #ffffff;
            font-weight: 500;
        }}

        /* Controls */
        .controls {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 8px;
            z-index: 100;
        }}

        .ctrl-btn {{
            width: 40px;
            height: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(15, 20, 50, 0.8);
            backdrop-filter: blur(12px);
            color: #c0c4d6;
            border-radius: 10px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.2s;
        }}

        .ctrl-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            transform: scale(1.05);
        }}

        /* Ambient glow */
        .glow {{
            position: fixed;
            width: 400px;
            height: 400px;
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.15;
            pointer-events: none;
            z-index: 0;
        }}

        .glow-1 {{
            top: -100px;
            right: -100px;
            background: #4CAF50;
        }}

        .glow-2 {{
            bottom: -100px;
            left: -100px;
            background: #2196F3;
        }}

        .glow-3 {{
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #FF6200;
            opacity: 0.08;
        }}

        /* Edge labels */
        .edge-label {{
            font-size: 9px;
            fill: #6b6f85;
            pointer-events: none;
        }}

        /* Node labels */
        .node-label {{
            font-size: 11px;
            fill: #c0c4d6;
            pointer-events: none;
            font-weight: 500;
        }}

        .node-label-target {{
            font-size: 13px;
            fill: #FFD700;
            font-weight: 700;
        }}

        /* Financial totals bar */
        .totals-bar {{
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }}

        .total-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 12px;
        }}

        .total-label {{
            color: #8b8fa3;
        }}

        .total-value {{
            font-weight: 600;
        }}

        .total-recv {{
            color: #4CAF50;
        }}

        .total-pay {{
            color: #F44336;
        }}

        .total-dep {{
            color: #2196F3;
        }}
    </style>
</head>
<body>
    <div class="glow glow-1"></div>
    <div class="glow glow-2"></div>
    <div class="glow glow-3"></div>

    <div class="sidebar">
        <h1>🌐 {company_name}</h1>
        <div class="subtitle">Commercial Network Graph</div>

        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_nodes', 0)}</div>
                <div class="stat-label">Nodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_edges', 0)}</div>
                <div class="stat-label">Connections</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item">
                <div class="legend-dot" style="background: #FFD700;"></div>
                <span>Target Company</span>
                <span class="legend-count">1</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #4CAF50;"></div>
                <span>Customers (120.xx)</span>
                <span class="legend-count">{stats.get('customer_count', 0)}</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #F44336;"></div>
                <span>Suppliers (320.xx)</span>
                <span class="legend-count">{stats.get('supplier_count', 0)}</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #2196F3;"></div>
                <span>Banks (102.xx)</span>
                <span class="legend-count">{stats.get('bank_count', 0)}</span>
            </div>
        </div>

        <div class="totals-bar">
            <div class="total-row">
                <span class="total-label">Total Receivables</span>
                <span class="total-value total-recv">₺{stats.get('total_receivables', 0):,.0f}</span>
            </div>
            <div class="total-row">
                <span class="total-label">Total Payables</span>
                <span class="total-value total-pay">₺{stats.get('total_payables', 0):,.0f}</span>
            </div>
            <div class="total-row">
                <span class="total-label">Bank Deposits</span>
                <span class="total-value total-dep">₺{stats.get('total_bank_deposits', 0):,.0f}</span>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip"></div>

    <div class="controls">
        <button class="ctrl-btn" onclick="zoomIn()" title="Zoom In">+</button>
        <button class="ctrl-btn" onclick="zoomOut()" title="Zoom Out">−</button>
        <button class="ctrl-btn" onclick="resetZoom()" title="Reset">⟲</button>
    </div>

    <div id="graph-container">
        <svg id="graph-svg"></svg>
    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        const nodesData = {nodes_json};
        const edgesData = {edges_json};

        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select('#graph-svg');
        const g = svg.append('g');

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.2, 5])
            .on('zoom', (event) => g.attr('transform', event.transform));
        svg.call(zoom);

        window.zoomIn = () => svg.transition().duration(300).call(zoom.scaleBy, 1.3);
        window.zoomOut = () => svg.transition().duration(300).call(zoom.scaleBy, 0.7);
        window.resetZoom = () => svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);

        // Arrow markers
        const defs = svg.append('defs');
        ['#4CAF50', '#F44336', '#2196F3', '#FFD700', '#999'].forEach(color => {{
            defs.append('marker')
                .attr('id', 'arrow-' + color.replace('#', ''))
                .attr('viewBox', '0 -5 10 10')
                .attr('refX', 20)
                .attr('refY', 0)
                .attr('markerWidth', 6)
                .attr('markerHeight', 6)
                .attr('orient', 'auto')
                .append('path')
                .attr('d', 'M0,-4L10,0L0,4')
                .attr('fill', color)
                .attr('opacity', 0.6);
        }});

        // Force simulation
        const simulation = d3.forceSimulation(nodesData)
            .force('link', d3.forceLink(edgesData)
                .id(d => d.id)
                .distance(d => {{
                    const w = d.weight || 1;
                    return Math.max(100, 300 - Math.log(w) * 20);
                }})
            )
            .force('charge', d3.forceManyBody().strength(-600))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(d => (d.size || 30) + 15))
            .force('x', d3.forceX(width / 2).strength(0.05))
            .force('y', d3.forceY(height / 2).strength(0.05));

        // Edges
        const link = g.append('g')
            .selectAll('line')
            .data(edgesData)
            .join('line')
            .attr('stroke', d => d.color || '#555')
            .attr('stroke-opacity', 0.35)
            .attr('stroke-width', d => Math.max(1.5, Math.min(5, Math.log(d.weight || 1) / 3)))
            .attr('marker-end', d => 'url(#arrow-' + (d.color || '#999').replace('#', '') + ')');

        // Edge labels
        const edgeLabel = g.append('g')
            .selectAll('text')
            .data(edgesData)
            .join('text')
            .attr('class', 'edge-label')
            .attr('text-anchor', 'middle')
            .text(d => d.label || '');

        // Node groups
        const node = g.append('g')
            .selectAll('g')
            .data(nodesData)
            .join('g')
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded)
            );

        // Outer glow
        node.append('circle')
            .attr('r', d => (d.size || 30) + 6)
            .attr('fill', 'none')
            .attr('stroke', d => d.color || '#999')
            .attr('stroke-opacity', 0.12)
            .attr('stroke-width', 4);

        // Main circle
        node.append('circle')
            .attr('r', d => d.size || 30)
            .attr('fill', d => d.color || '#999')
            .attr('fill-opacity', d => d.type === 'target' ? 0.9 : 0.7)
            .attr('stroke', d => d.color || '#999')
            .attr('stroke-width', 2)
            .attr('stroke-opacity', 0.8)
            .style('cursor', 'pointer')
            .style('filter', d => d.type === 'target' ? 'drop-shadow(0 0 12px rgba(255,215,0,0.5))' : 'none');

        // Labels
        node.append('text')
            .attr('class', d => d.type === 'target' ? 'node-label-target' : 'node-label')
            .attr('dy', d => (d.size || 30) + 16)
            .attr('text-anchor', 'middle')
            .text(d => {{
                const label = d.label || d.id;
                return label.length > 20 ? label.substring(0, 18) + '…' : label;
            }});

        // Tooltip
        const tooltip = d3.select('#tooltip');

        node.on('mouseenter', (event, d) => {{
            const typeLabel = {{
                'target': '🏢 Target Company',
                'customer': '🤝 Customer (Alıcı)',
                'supplier': '🏭 Supplier (Satıcı)',
                'bank': '🏦 Bank (Banka)',
            }}[d.type] || d.type;

            tooltip.html(`
                <div class="tt-label">${{d.label || d.id}}</div>
                <div class="tt-row"><span class="tt-key">Type</span><span class="tt-val">${{typeLabel}}</span></div>
                ${{d.account_code ? `<div class="tt-row"><span class="tt-key">Hesap Kodu</span><span class="tt-val">${{d.account_code}}</span></div>` : ''}}
                ${{d.balance ? `<div class="tt-row"><span class="tt-key">Balance</span><span class="tt-val">₺${{d.balance.toLocaleString('tr-TR')}}</span></div>` : ''}}
            `);

            tooltip.classed('visible', true)
                .style('left', (event.clientX + 16) + 'px')
                .style('top', (event.clientY - 10) + 'px');
        }})
        .on('mousemove', (event) => {{
            tooltip.style('left', (event.clientX + 16) + 'px')
                .style('top', (event.clientY - 10) + 'px');
        }})
        .on('mouseleave', () => {{
            tooltip.classed('visible', false);
        }});

        // Hover highlight
        node.on('mouseenter.highlight', (event, d) => {{
            const connectedIds = new Set();
            connectedIds.add(d.id);
            edgesData.forEach(e => {{
                const src = typeof e.source === 'object' ? e.source.id : e.source;
                const tgt = typeof e.target === 'object' ? e.target.id : e.target;
                if (src === d.id) connectedIds.add(tgt);
                if (tgt === d.id) connectedIds.add(src);
            }});

            node.select('circle:nth-child(2)')
                .transition().duration(200)
                .attr('fill-opacity', n => connectedIds.has(n.id) ? 0.9 : 0.15);

            node.select('text')
                .transition().duration(200)
                .attr('fill-opacity', n => connectedIds.has(n.id) ? 1 : 0.2);

            link.transition().duration(200)
                .attr('stroke-opacity', e => {{
                    const src = typeof e.source === 'object' ? e.source.id : e.source;
                    const tgt = typeof e.target === 'object' ? e.target.id : e.target;
                    return (src === d.id || tgt === d.id) ? 0.8 : 0.05;
                }});

            edgeLabel.transition().duration(200)
                .attr('fill-opacity', e => {{
                    const src = typeof e.source === 'object' ? e.source.id : e.source;
                    const tgt = typeof e.target === 'object' ? e.target.id : e.target;
                    return (src === d.id || tgt === d.id) ? 1 : 0.1;
                }});
        }})
        .on('mouseleave.highlight', () => {{
            node.select('circle:nth-child(2)')
                .transition().duration(300)
                .attr('fill-opacity', d => d.type === 'target' ? 0.9 : 0.7);

            node.select('text')
                .transition().duration(300)
                .attr('fill-opacity', 1);

            link.transition().duration(300)
                .attr('stroke-opacity', 0.35);

            edgeLabel.transition().duration(300)
                .attr('fill-opacity', 1);
        }});

        // Simulation tick
        simulation.on('tick', () => {{
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);

            edgeLabel
                .attr('x', d => (d.source.x + d.target.x) / 2)
                .attr('y', d => (d.source.y + d.target.y) / 2 - 6);

            node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
        }});

        // Drag
        function dragStarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}

        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}

        function dragEnded(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}

        // Initial zoom to fit
        setTimeout(() => {{
            svg.transition().duration(800)
                .call(zoom.transform, d3.zoomIdentity.translate(width * 0.15, 0).scale(0.85));
        }}, 1000);
    </script>
</body>
</html>"""

    logger.info(f"Generated interactive network graph HTML: {len(nodes)} nodes, {len(edges)} edges")
    return html


def save_network_html(network_data: dict, output_dir: str, company_name: str) -> str:
    """
    Generate and save the interactive network graph HTML file.

    Args:
        network_data: dict with 'nodes', 'edges', 'stats'
        output_dir: Output directory path
        company_name: Company name for filename

    Returns:
        Path to the saved HTML file
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
    filename = f"graph_{safe_name}.html"
    filepath = os.path.join(output_dir, filename)

    html_content = generate_network_html(network_data, company_name)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Saved network graph: {filepath}")
    return filepath

# Traffic Guardian - Draw.io XML Code for DFD Levels 1 & 2

Below are the XML code blocks for DFD Level 1, DFD Level 2 (Preprocessing Pipeline), and DFD Level 2 (Inference & ER Entity Interactions) that you can copy/paste directly into Draw.io (app.diagrams.net).

To import any diagram into Draw.io:
1. Open Draw.io (app.diagrams.net).
2. Go to **Arrange** > **Insert** > **Advanced** > **XML...**
3. Paste the XML block and click **Insert**.

---

## 1. DFD Level 1 (System-Wide Data Flow)

```xml
<mxGraphModel dx="1200" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="600">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="e1" value="Browser" style="shape=rect;rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="20" y="220" width="100" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="p1" value="1.0&#xa;Data Collection&#xa;(Extension)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="180" y="210" width="150" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="p2" value="2.0&#xa;Feature Transform&#xa;(Backend)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="390" y="210" width="150" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="p3" value="3.0&#xa;Model Inference" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="600" y="210" width="150" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="p4" value="4.0&#xa;Ensemble&#xa;(Layer-2)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;" vertex="1" parent="1">
      <mxGeometry x="810" y="210" width="130" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="p5" value="5.0&#xa;Result Display&#xa;(Popup)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;" vertex="1" parent="1">
      <mxGeometry x="810" y="80" width="130" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="e2" value="User" style="shape=rect;rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="1020" y="90" width="80" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="f1" value="Raw HTTP&#xa;metadata" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="e1" target="p1" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f2" value="Session JSON" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="p1" target="p2" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f3" value="X_lstm, X_resnet, X_xgb" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="p2" target="p3" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f4" value="probs (1,2) x3" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="p3" target="p4" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f5" value="verdict" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="p4" target="p5" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f6" value="Badge/Alert" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="p5" target="e2" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
  </root>
</mxGraphModel>
```

---

## 2. DFD Level 2 (Offline Preprocessing Pipeline)

```xml
<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="800">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="d1" value="D1: packet_based&#xa;trainset.csv&#xa;(4.4M x 25)" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="30" y="30" width="160" height="60" as="geometry"/>
    </mxCell>
    <mxCell id="d2" value="D2: session_based&#xa;trainset.csv&#xa;(488K x 280)" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="30" y="130" width="160" height="60" as="geometry"/>
    </mxCell>
    <mxCell id="p1" value="2.1&#xa;Merge on&#xa;unique_link_mark" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="280" y="60" width="160" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="p2" value="2.2&#xa;LSTM Tensor Builder&#xa;groupby→cutoff→pad&#xa;→(N,15,85)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="510" y="60" width="180" height="80" as="geometry"/>
    </mxCell>
    <mxCell id="p3" value="2.3&#xa;Session Alignment&#xa;by uid_list" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="510" y="200" width="170" height="65" as="geometry"/>
    </mxCell>
    <mxCell id="p4" value="2.4&#xa;ResNet: SelectKBest&#xa;→MinMax→outer_product&#xa;→(N,1,38,38)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="510" y="320" width="190" height="80" as="geometry"/>
    </mxCell>
    <mxCell id="p5" value="2.5&#xa;XGBoost: 65 ratio&#xa;+32 ENC = 97 cols&#xa;→(N,97)" style="ellipse;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="510" y="450" width="190" height="80" as="geometry"/>
    </mxCell>
    <mxCell id="d3" value="D3: tensors/*.npy" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="810" y="320" width="150" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="d4" value="D4: scalers/*.pkl" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="810" y="380" width="150" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="d5" value="D5: uid_list.npy" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="810" y="440" width="150" height="40" as="geometry"/>
    </mxCell>
    <mxCell id="f1" edge="1" source="d1" target="p1" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f2" edge="1" source="d2" target="p1" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f3" value="merged (4.4M x 103)" style="endArrow=block;endFill=1;fontSize=9;" edge="1" source="p1" target="p2" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f4" value="uid_list" style="endArrow=block;endFill=1;fontSize=9;" edge="1" source="p2" target="p3" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f5" value="aligned sess" style="endArrow=block;endFill=1;fontSize=9;" edge="1" source="p3" target="p4" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f6" edge="1" source="p3" target="p5" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f7" edge="1" source="p4" target="d3" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f8" edge="1" source="p5" target="d3" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f9" edge="1" source="p2" target="d3" parent="1">
      <mxGeometry relative="1" as="geometry">
        <Array as="points">
          <mxPoint x="780" y="100"/>
        </Array>
      </mxGeometry>
    </mxCell>
  </root>
</mxGraphModel>
```

---

## 3. DFD Level 2 (Inference & ER Entity Interactions)

```xml
<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1000">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="ext1" value="Browser&#xa;Extension" style="shape=rect;rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="20" y="220" width="100" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="proc21" value="2.1&#xa;Capture &amp;&#xa;Sessionize" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="190" y="210" width="140" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="db_packet" value="D1: Packet Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="400" y="220" width="130" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="db_session" value="D2: Session Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="400" y="120" width="130" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="proc22" value="2.2&#xa;Scaler &amp;&#xa;Feature Transform" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="600" y="210" width="140" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="db_settings" value="D7: Settings Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="600" y="20" width="140" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="db_tensor" value="D6: Tensor Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="600" y="380" width="140" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="proc23" value="2.3&#xa;Neural &amp;&#xa;Ensemble Inference" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="810" y="210" width="140" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="db_model" value="D5: Model Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="810" y="380" width="140" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="proc24" value="2.4&#xa;Logging &amp;&#xa;Audit" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1">
      <mxGeometry x="1020" y="210" width="140" height="70" as="geometry"/>
    </mxCell>
    <mxCell id="db_prediction" value="D3: Prediction Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="1020" y="20" width="140" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="db_history" value="D4: HistoryEntry Table" style="shape=parallelogram;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontSize=10;" vertex="1" parent="1">
      <mxGeometry x="1020" y="380" width="140" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="ext2" value="User&#xa;Dashboard" style="shape=rect;rounded=0;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="1230" y="220" width="100" height="50" as="geometry"/>
    </mxCell>
    <mxCell id="f21" value="HTTP packet events" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="ext1" target="proc21" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f22" value="Insert Packets" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc21" target="db_packet" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f23" value="Create Session" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc21" target="db_session" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f24" value="Read Session" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="db_session" target="proc22" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f25" value="Fetch Packets" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="db_packet" target="proc22" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f26" value="Read Configs" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="db_settings" target="proc22" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f27" value="Read Scaler Paths" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="db_tensor" target="proc22" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f28" value="Structured Tensors" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc22" target="proc23" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f29" value="Load Model weights" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="db_model" target="proc23" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f30" value="Branch probs" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc23" target="proc24" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f31" value="Insert prediction" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc24" target="db_prediction" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f32" value="Insert trace" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc24" target="db_history" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
    <mxCell id="f33" value="Output classification" style="endArrow=block;endFill=1;html=1;fontSize=9;" edge="1" source="proc24" target="ext2" parent="1">
      <mxGeometry relative="1" as="geometry"/>
    </mxCell>
  </root>
</mxGraphModel>
```

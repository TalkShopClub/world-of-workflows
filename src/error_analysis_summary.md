# State Prediction Error Analysis Summary

## Overview

This analysis examines errors in state prediction across multiple models using IoU (Intersection over Union) metrics. The analysis reveals significant challenges in accurately predicting ServiceNow state changes.

## Key Findings

### 1. Overall Performance Metrics

| Model | SysAudit IoU | Additional Info IoU | Full Accuracy | Avg Side Effects/File |
|-------|--------------|-------------------|---------------|----------------------|
| GPT-5 | 0.194 | 0.486 | 0.004 | 13.06 |
| Claude-Sonnet-4.5 | 0.170 | 0.386 | 0.000 | 17.91 |
| Gemini-2.5-Pro | 0.224 | 0.418 | 0.005 | 21.37 |

### 2. Critical Issues Identified

#### A. Very Low Full Accuracy
- **GPT-5**: 0.4% full accuracy
- **Claude-Sonnet-4.5**: 0.0% full accuracy  
- **Gemini-2.5-Pro**: 0.5% full accuracy

This indicates that models almost never predict the complete state change correctly.

#### B. Low SysAudit IoU
- All models show SysAudit IoU between 0.17-0.22
- This means only ~20% overlap between predicted and actual audit records
- Significant room for improvement in core state prediction

#### C. High Side Effects
- Average of 13-21 side effects per file
- Side effects increase with K value (more steps = more errors)
- Indicates models are predicting many changes that don't actually occur

### 3. Most Common Error Patterns

#### Top Side Effects (False Positives):
1. **sys_user.last_login_time** - 356-832 occurrences
2. **metric_instance.start** - 48-98 occurrences  
3. **incident.state** - 32-37 occurrences
4. **incident.short_description** - 29 occurrences
5. **metric_instance.id/table/field/value** - 28-49 occurrences

#### Key Observations:
- **Temporal Fields**: `last_login_time` is the most problematic field across all models
- **Metric Instances**: Models struggle with metric-related table predictions
- **Incident Fields**: Basic incident fields are frequently mispredicted

### 4. Error Analysis by K Value

Side effects increase significantly with prediction horizon:

| K Value | GPT-5 | Claude-4.5 | Gemini-2.5-Pro |
|---------|-------|------------|----------------|
| K=1 | 245 | 317 | 355 |
| K=2 | 389 | 505 | 596 |
| K=3 | 489 | 671 | 815 |
| K=4 | 606 | 844 | 969 |
| K=5 | 687 | 976 | 1112 |

**Pattern**: Error accumulation increases linearly with prediction steps.

### 5. Detailed Error Categories

#### A. Action Prediction Errors
- **Tool Name Mismatches**: Models sometimes predict wrong ServiceNow tools
- **Parameter Mismatches**: Incorrect parameter values in action calls

#### B. State Prediction Errors
- **Missing Fields**: Ground truth fields not predicted (false negatives)
- **Extra Fields**: Predicted fields not in ground truth (false positives)
- **Value Mismatches**: Correct field predicted but wrong values

#### C. Schema-Related Issues
- **Invalid Tables**: Predictions include non-existent ServiceNow tables
- **Invalid Fields**: Field names that don't exist in the schema
- **Type Mismatches**: Wrong data types for boolean/date fields

### 6. Model-Specific Insights

#### GPT-5
- Best overall SysAudit IoU (0.194)
- Lowest side effects per file (13.06)
- Most balanced performance across metrics

#### Claude-Sonnet-4.5
- Lowest SysAudit IoU (0.170)
- Highest side effects per file (17.91)
- Struggles most with over-prediction

#### Gemini-2.5-Pro
- Highest SysAudit IoU (0.224)
- Highest total side effects (21.37 per file)
- Best at core prediction but worst at avoiding false positives

### 7. Root Cause Analysis

#### A. Temporal Understanding
- Models struggle with timestamp fields (`last_login_time`)
- Inconsistent date/time format handling
- Poor understanding of temporal relationships

#### B. Schema Awareness
- Limited knowledge of ServiceNow table structure
- Inability to distinguish between valid and invalid fields
- Poor understanding of field types and constraints

#### C. Context Understanding
- Models don't understand which fields should change for specific actions
- Over-prediction of unrelated system fields
- Under-prediction of core business logic fields

#### D. Action-State Relationship
- Weak connection between predicted actions and resulting state changes
- Models predict actions but struggle to predict their consequences
- Poor understanding of ServiceNow workflow behavior

### 8. Recommendations

#### Immediate Improvements
1. **Temporal Field Handling**: Implement specialized logic for timestamp fields
2. **Schema Validation**: Add validation against ServiceNow schema
3. **Field Filtering**: Reduce over-prediction of system fields
4. **Action-Context Mapping**: Better understanding of which fields change for specific actions

#### Medium-term Improvements
1. **ServiceNow Knowledge**: Enhanced training on ServiceNow-specific patterns
2. **Workflow Understanding**: Better modeling of ServiceNow workflow behavior
3. **Error Correction**: Post-processing to fix common error patterns
4. **Ensemble Methods**: Combine multiple models for better accuracy

#### Long-term Improvements
1. **Fine-tuning**: Domain-specific fine-tuning on ServiceNow data
2. **Retrieval-Augmented Generation**: Use ServiceNow documentation for better context
3. **Multi-step Reasoning**: Better planning for multi-step predictions
4. **Human-in-the-loop**: Incorporate human feedback for error correction

### 9. Success Metrics for Improvement

#### Target Metrics
- **SysAudit IoU**: Increase from ~0.2 to >0.5
- **Full Accuracy**: Increase from ~0.0% to >10%
- **Side Effects**: Reduce from 13-21 to <5 per file
- **Temporal Accuracy**: >80% accuracy on timestamp fields

#### Validation Approach
- A/B testing with improved models
- Human evaluation of critical predictions
- Business impact assessment of prediction accuracy

## Conclusion

The state prediction task presents significant challenges, with current models achieving very low accuracy. The primary issues are:

1. **Over-prediction** of system fields (especially temporal ones)
2. **Under-prediction** of core business logic fields
3. **Poor schema awareness** and validation
4. **Weak action-state relationship** understanding

Addressing these issues through targeted improvements in temporal handling, schema validation, and context understanding could significantly improve prediction accuracy.


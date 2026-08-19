# Security Incident Response Plan

## 1. Purpose

This plan defines how the organization detects, classifies, escalates, and reports security
incidents, including personal data breaches.

## 2. Severity Classification

Severity 1 (Critical): confirmed unauthorized access to Personal Data, ransomware affecting
production, or loss of a production system with no available recovery point.
Severity 2 (High): suspected unauthorized access, a single-system compromise contained to
non-production, or degradation of a security control on a Critical system.
Severity 3 (Moderate): policy violations, isolated malware detections cleaned by endpoint tooling,
or a failed control test with no evidence of exploitation.

## 3. Escalation Timelines

Severity 1 incidents must be escalated to the Incident Commander within fifteen (15) minutes of
detection, and to executive leadership within one (1) hour. Severity 2 incidents must be escalated
within four (4) hours. Severity 3 incidents must be logged within one (1) business day.

## 4. Regulatory Notification

Where an incident constitutes a personal data breach, the Data Protection Officer must notify the
relevant supervisory authority within seventy-two (72) hours of becoming aware of the breach. Where
notification exceeds seventy-two hours, the notification must be accompanied by the reasons for the
delay. Affected individuals must be notified without undue delay where the breach is likely to
result in a high risk to their rights and freedoms.

## 5. Evidence Handling

All evidence must be collected in a manner that preserves its integrity, with a documented chain of
custody. Volatile data must be captured before a system is powered down. Forensic images must be
hashed at acquisition, and the hash recorded in the incident record.

## 6. Post-Incident Review

A post-incident review must be completed within ten (10) business days of incident closure for all
Severity 1 and Severity 2 incidents. The review must identify root cause, contributing factors, and
corrective actions with named owners and due dates. Corrective actions are tracked to completion by
the Security Governance function.

## 7. Testing

The incident response plan must be tested at least annually through a tabletop exercise or
simulation. Test results and identified gaps must be documented and remediated.

% NOTE: property value declarations and choice rules are generated from
% PROPERTIES in domain.py — do not hardcode them here.

% ── Region capacity (at most 3 objects per region) ────────────────────────
%:- #count { X : object(X), hasProperty(X, region, R) } >= 4, property(region, R).

% ── Region spatial layout (2×2 grid) ─────────────────────────────────────
% r0=top-left  | r1=top-right
% r2=bot-left  | r3=bot-right
right_R(r1, r0). right_R(r3, r0).
right_R(r1, r2). right_R(r3, r2).
left_R(R1, R2) :- right_R(R2, R1).

front_R(r2, r0). front_R(r3, r0).
front_R(r2, r1). front_R(r3, r1).
behind_R(R1, R2) :- front_R(R2, R1).

% ── Spatial relationships derived from region placement ───────────────────
hasRelationship(X, Y, left)   :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), left_R(R1, R2).
hasRelationship(X, Y, right)  :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), right_R(R1, R2).
hasRelationship(X, Y, front)  :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), front_R(R1, R2).
hasRelationship(X, Y, behind) :- object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2), behind_R(R1, R2).

% Spatial relationships among objects: disjunctive left/right and front/behind
hasRelationship(X, Y, left); hasRelationship(X, Y, right) :-
    object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2).
hasRelationship(X, Y, front); hasRelationship(X, Y, behind) :-
    object(X), object(Y), X != Y,
    hasProperty(X, region, R1), hasProperty(Y, region, R2).

% Anti-symmetry
:- hasRelationship(X1, X2, R), hasRelationship(X2, X1, R), X1 != X2.

% Symmetry
hasRelationship(X1, X2, left)   :- hasRelationship(X2, X1, right).
hasRelationship(X1, X2, front)  :- hasRelationship(X2, X1, behind).
hasRelationship(X1, X2, right)   :- hasRelationship(X2, X1, left).
hasRelationship(X1, X2, behind)  :- hasRelationship(X2, X1, front).


% Transitivity
hasRelationship(X1, X3, left)   :- hasRelationship(X1, X2, left),
    hasRelationship(X2, X3, left),   X1 != X2, X2 != X3.
hasRelationship(X1, X2, left)   :- hasRelationship(X2, X1, right).
hasRelationship(X1, X3, behind) :- hasRelationship(X1, X2, behind),
    hasRelationship(X2, X3, behind), X1 != X2, X2 != X3.
hasRelationship(X1, X2, behind) :- hasRelationship(X2, X1, front).

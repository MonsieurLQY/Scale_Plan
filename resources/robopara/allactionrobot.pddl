(define (domain desktop_cleanup)
 (:requirements :strips :typing :negative-preconditions)
 (:types robot object state)
 (:predicates
  (handempty ?r - robot) (holding ?r - robot ?o - object)
  (at-location ?o - object ?s - object) (reachable ?r - robot ?o - object)
  (support-ready ?s - object) (closed ?o - object) (upright ?o - object))
 (:action pick
  :parameters (?robot - robot ?source - object ?target - object)
  :precondition (and (handempty ?robot) (at-location ?target ?source) (reachable ?robot ?target))
  :effect (and (holding ?robot ?target) (not (handempty ?robot)) (not (at-location ?target ?source))))
 (:action place
  :parameters (?robot - robot ?source - object ?target - object)
  :precondition (and (holding ?robot ?source) (support-ready ?target) (reachable ?robot ?target))
  :effect (and (at-location ?source ?target) (handempty ?robot) (not (holding ?robot ?source))))
 (:action handover
  :parameters (?donor - robot ?receiver - robot ?object - object)
  :precondition (and (holding ?donor ?object) (handempty ?receiver))
  :effect (and (holding ?receiver ?object) (handempty ?donor) (not (holding ?donor ?object)) (not (handempty ?receiver))))
 (:action flap_close
  :parameters (?robot - robot ?object - object)
  :precondition (and (handempty ?robot) (reachable ?robot ?object))
  :effect (and (closed ?object) (support-ready ?object)))
 (:action adjust
  :parameters (?robot - robot ?object - object ?target_state - state)
  :precondition (and (handempty ?robot) (reachable ?robot ?object))
  :effect (upright ?object))
)

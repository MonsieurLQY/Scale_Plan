(define (domain robot23 )
  (:requirements :strips :typing :negative-preconditions) 
  (:types robot object)
  
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot) 
    (at-location  ?object - object ?location - object)
    (object-open ?robot - robot ?object - object)
    (object-close ?robot - robot ?object - object)
  )
  
  (:action GoToObject
    :parameters (?robot - robot ?object - object)
    :precondition (not (inaction ?robot))
    :effect (and 
      (at ?robot ?object)
      (forall (?another_object - object)
        (when (at ?robot ?another_object)
          (not (at ?robot ?another_object))
        )
      )
      (not (inaction ?robot))
    )
  )
  
    (:action OpenObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-open ?robot ?object)
    )
  )

  (:action CloseObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
      (not (inaction ?robot))
      (at ?robot ?object)
    )
    :effect (and
      (not (inaction ?robot))
      (object-close ?robot ?object)
    )
  )
)
(define (domain robot26 )
  (:requirements :strips :typing :negative-preconditions) 
  (:types robot object)
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot) 
    (holding ?robot - robot ?object - object)
    (at-location  ?object - object ?location - object)
    (sliced ?object - object)
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

  (:action PickupObject      
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and 
      (at-location ?object ?location)
      (at ?robot ?location)
      (not (holding ?robot ?object))
      (not (inaction ?robot))
    )
    :effect (and
      (holding ?robot ?object)
      (not (inaction ?robot))
    )
  )

  (:action SliceObject
    :parameters (?robot - robot ?object - object ?location - object ?knife - object)
    :precondition (and 
      (at-location ?object ?location)
      (at ?robot ?location)
      (not (inaction ?robot))
      (holding ?robot ?knife)
    )
    :effect (and
      (not (inaction ?robot))
      (sliced ?object)
    )
  )    
)